# Paper IV (molecular-vibronic H2O) worklog

Deliverable file for Paper-IV findings and decisions. Update in place.

## CHTC status audit — 2026-09-02

Sources: `~/chtc_rescue/` (143 GB read-only mirror of the CHTC submit node,
captured 2026-08-31T21:50 CDT), its `scheduler/condor_q` + `condor_history`
captures, and the repo-local `chtc/paper_iv_h2o_*` submission records.

### Bottom line

The **static SNAKE application run finally executes**. Cluster **9715945**
(`paper_iv_h2o_append_proxy_373_d50_20260829`) was `R` — running, healthy — at
capture: 1 d 09 h 37 m elapsed, resident memory flat at 3.96 GB against a
32 GB request, no eviction, no hold, no restart.

**The `+MaxRuntime = 259200` in `submit.sub` is not enforced.** It is a bare
ClassAd attribute and the submit file carries no `periodic_remove` expression
that consumes it, so nothing terminates the job at 72 h. An earlier reading of
this worklog treated 2026-09-02 12:13:53 CDT as a hard wall; that was wrong.

Confirmed 2026-09-02 18:10 CDT by rsync against the submit node:
`/home/jsstrobel/chtc/d50pkg/results/` **does not exist**. Since
`when_to_transfer_output = ON_EXIT_OR_EVICT` only materializes `results/` when
the job leaves the machine, its absence is consistent with the job still
running — now ~3 d 06 h elapsed.

### The physics that succeeded first (cluster 9705318, 08-26, 3.1 h)

Cutoff-convergence validation of the finite active-space, three-mode, linearly
coupled H2O Hamiltonian. This is the `(3,7,3)/19q` converged model the
`f433b89d` commit message refers to.

| Quantity | Value |
|---|---|
| Working cutoffs `n_ph_work` (sym. stretch, bend, antisym. stretch) | (3, 7, 3) |
| Reference cutoffs | (4, 8, 4) |
| Total qubits | 19 |
| Sector dimension | 28,800 |
| `E_work` | −75.014463513376 Ha |
| `E_ref` | −75.015134943251 Ha |
| `abs(ΔE)` | 6.7143e−4 Ha (tol 1.6e−3) — **pass** |
| Total boundary weight | 6.2066e−3 (tol 1e−2) — **pass** |
| Per-mode boundary weight | sym 5.028e−3 · bend 9.778e−4 · antisym 2.093e−4 |
| Elapsed | 11,176 s |

**This 28 MB fixture is not in the repo and not git-tracked.** It exists only at
`~/chtc_rescue/home/chtc/paper_iv_h2o_reencode_20260826/results/h2o_linear_fd_work373_ref484_fixture_20260826.json`
(summary beside it as `reencode_summary_work373_ref484.json`). The repo
references the model only as a comment,
`pipelines/static_adapt/ra_adapt/pools.py:300`. It is the sole input to the d50
run and the only cutoff-convergence evidence Paper IV has.

### Full attempt ledger

| Cluster | Date | Campaign | Outcome |
|---|---|---|---|
| 8887458 | — | SR d50 | worker executable relocation / self-hash path failure; no science started |
| 8973939 | — | SR d50 | H2O derivative-pool controller compatibility omission; exit 1 |
| 9378923 | 07-24 | `sr_derivative_resolved_v2_d50` | wrong scientific route (metric prune + beam); removed by user before start |
| 9378925 | 07-24 | v3 route | runtime profile registration plumbing; exit 1 after 21 s |
| 9379290 | 07-25 | `sr_paper_i_noprune_nobeam_d50` | idle at recording; aged out of the 08-31 history window — no completion record, no output dir |
| 9662568 | 08-18 | `reencode_20260818` at (3,6,3)/(3,5,3) | ran 21–24 min, then `condor_rm` by user 08-27 |
| 9705182 | 08-26 | reencode | **held — cgroup memory limit**, 12,288 MB requested, 12,026 MB measured; removed |
| 9705308 | 08-26 | reencode | removed after 81 s |
| 9705318 | 08-26 | reencode work373/ref484 | **COMPLETED, passed** (table above) |
| 9715899 | 08-30 | d50 | removed 27 min after submit, 0 s runtime |
| **9715945** | **08-30** | **d50 append-only** | **RAN 1 d 09 h+, healthy; 72 h wall hit 09-02 12:13 CDT; results unfetched** |

So: the memory-limit failure at 9705182 was repaired by raising the request; the
route/plumbing failures of July were repaired by the centralized runtime profile
registry; and `f433b89d` (engine forced plateau, pool gate required equal
cutoffs) is what made this append-only, per-mode-cutoff run reachable at all.

### Run contract actually executing (9715945)

`python3 -m pipelines.static_adapt.ra_adapt.h2o_application` with
`--fixture h2o_linear_fd_work373_ref484_fixture_20260826.json`,
`--maximum-controller-rounds 50`, `--insertion append_only`,
`--checkpoint-history-tail 1`, `--no-estimator-ledger`; single-threaded BLAS,
disk-backed sector-audit cache, 19-qubit Pauli sign cache. 4 CPUs / 32 GB /
30 GB disk.

### POST-MORTEM — 9715945 output is unrecoverable (confirmed 2026-09-03)

The job was held at **2026-09-02 12:13:52** by a CHTC pool policy:
`Job failed to complete in 72 hrs`, hold code 21 subcode 0, after
`RemoteWallClockTime = 252,164 s` (70.0 h) and `TimeSlotBusy = 259,201 s`.

**All output is lost.** Verified on the submit node:

| Probe | Result |
|---|---|
| `condor_q 9715945 -af StageOutStart StageOutFinish SpooledOutputFiles` | all **undefined** — stage-out never began |
| `/var/lib/condor/spool/5945/0/cluster9715945.proc0.subproc0` | does not exist |
| `condor_transfer_data 9715945` | fails: `reading from file /home/jsstrobel/chtc/d50pkg/results: (errno 2)` |
| `/home/jsstrobel/chtc/d50pkg/` | only `logs/`, `run_d50.sh`, `source.tgz`, `submit.sub` |
| `/staging/jsstrobel/` | no d50/h2o/paper_iv entry |
| `condor_ssh_to_job` | refused — job on hold, scratch reclaimed |

**Root cause.** CHTC's 72 h limit is a `SYSTEM_PERIODIC_HOLD`, not a preemption.
`when_to_transfer_output = ON_EXIT_OR_EVICT` returns the sandbox on normal exit
and on *graceful eviction*; a policy hold terminates with no stage-out. The
engine was writing a complete resume checkpoint to `results/current.json`
atomically after **every** controller round
(`h2o_application.py:171-175`, `adapt_pipeline.py:21488`) — onto execute-node
scratch that was never transferred and is now reclaimed.

The July campaign recorded `"periodic_checkpoint_transfer": true` in
`chtc/paper_iv_h2o_sr_derivative_resolved_v2_d50_20260718/submission_record.json`.
The live `d50pkg/submit.sub` carries no checkpoint transfer, no
`periodic_remove`, and no `max_retries`. Dropping that pattern is what cost the
run. The bare `+MaxRuntime = 259200` in the submit file is inert and enforces
nothing; the wall is imposed by the pool.

Memory and disk were never the constraint: 4,151 MB of a 32 GB request (12.1%,
plateaued at 65 min, creeping 34 MB/day), and 300,000 KB of a 30 GB disk request
(1.0%). Only wall-clock mattered, and 50 controller rounds did not fit in 72 h.

### Required before any resubmission

1. **`--resume-checkpoint` in `h2o_application.py`.** The entrypoint defines 10
   argparse options, none a resume, and calls `run_ra_adapt(problem, protocol)`
   at `:309` with no operational controls — hard-coded `FreshStart`. The engine
   side is complete: `RAAdaptOperationalControls(resume=AcceptedStateResume(
   checkpoint_path, checkpoint_sha256))` (`ra_adapt/contracts.py:280-296`,
   `sr_snake/contracts.py:687-701`) and it supports `append_only`. The only
   in-repo caller that reaches it, `campaign.py:1365`, is hard-locked to
   `problem_key="hh"`. Resume also requires the digest-matched estimator-ledger
   sidecar beside `current.json` (`sr_snake/_resume.py:4047-4059`).
2. **Segmented submission.** Each job runs a bounded round count and exits
   normally so output transfers on exit; the next segment resumes from the
   previous `current.json`. Target well under the 72 h pool wall.
3. **Never re-run into a `results/` that already holds `current.json`.** Only
   `result.json` is guarded (`h2o_application.py:243-247`); a restart silently
   overwrites the checkpoint at its own round 1.
4. **Track the fixture.** `h2o_linear_fd_work373_ref484_fixture_20260826.json`
   (28 MB) still exists only under `~/chtc_rescue/` and on the submit node.


## Build round — 2026-09-03 (campaign branch `paper-iv/h2o-campaign-20260903`)

Base decision: integrate the b3 speed-up **into main**, not the reverse. b3 is
679 commits behind main, lacks `f433b89d` and 13 `ra_adapt`/`sr_snake`
refactors; picking `9a43bfe2` onto it conflicted immediately.

| Commit | Lane | Acceptance evidence |
|---|---|---|
| `d6d44af3`, `3b1b86f4` | b3 chunked encoder + opt-in `STATIC_ADAPT_CHECKPOINT_JSON_LAYOUT=compact` (cherry-picked) | 61 passed |
| `34c87900` | port of `d893a6b0` O(1) ledger (b3's third commit conflicted with main) | ledgers byte-identical pristine vs ported; 51 differing keys = 36 run-to-run noise + 15 source-provenance digests; `to_payload` walks per 3-round leg 33 → 5; 20 passed, 6 skipped (streaming absent on main); 18 route-suite failures identical on pristine |
| `929f1b47` | `--max-wall-seconds` as `RAAdaptOperationalControls.maximum_wall_seconds`, stop reason `wall_clock_budget_v1`, checked only after a completed round, lowest precedence; protocol sha unchanged | opt-in CLI: `--max-wall-seconds 0` → exit 0, 1 round, `current.json` + sha-matched sidecar + `result.json`; `protocol.json` byte-identical |
| `929f1b47` (same lane) | **pre-existing crash fixed**: every H2O run on main exits 1 after its last round (`runtime.py:1947` → `paper_i_run_summary.py:1630`, from `90b19f3f`) — H2O family now skips the Paper-I canonical summary it must not inherit | unmodified: exit 1 after 339 s; patched: exit 0 |
| `af222b69` | 19q FakeMarrakesh embedding `(0,1,2,3,4,16,5,23,6,22,7,21,8,20,9,24,10,25,11)` — nested-prefix, tree, rule regenerates all six historical entries; `--cost-mode {proxy,marrakesh_graph_span_v1}` via typed `MarrakeshGraphSpanCompileCost`, suffix `__cost-marrakesh_graph_span_v1`; default sha pinned unchanged | 14 + 18 passed; graph-span rejected off H2O |
| `fd37ec19` | fixture tracked at `test_support/paper_iv_h2o/` (sha `190387449b…`, differs from July `570690bd…` by lineage); compact-layout resume proven through the production writers; two-arm package `chtc/paper_iv_h2o_two_arm_20260903/` | 7 + 7 passed; package builds, preflight blocked only on flags that have since merged |
| `9a576e4a` | merge of the three lanes | **130 passed, 9 skipped** |

Graph-span cost at 19q, measured not optimized: the Steiner DP is exact
Dreyfus–Wagner over 156 physical qubits, ~3× per extra terminal (k=12 5.3 s,
13 15.5 s, 14 44 s). The (3,7,3) pool has 2,542 unique supports (max 15);
pricing all once ≈ 27 min per process, cache per process. The embedding's one
ambiguous step (logical index 15 at dead-end qubit 20) was resolved toward the
two-row band; alternatives are in the code comment for the user to overrule.

### Blocker found by two lanes independently — now the active build

`--resume-checkpoint` on the H2O route is refused at
`sr_snake/_resume.py:1455` (`_validate_problem_binding`: canonical L=2 HH or
the pure-Hubbard noise application only), before the sidecar-sha check;
`:1471-1472` rejects any request carrying the H2O fixture; `:1543-1556` pins
`problem ∈ {hh, hubbard}`, `adapt_pool = full_meta`. This is the lane
contract's named deferred item. Design in progress: a third named application
branch with **fixture-sha problem binding** (path strings are not identity),
H2O required settings derived from the canonical H2O v3 contract, proxy ↔
graph-span refusal kept, HH/Hubbard branches untouched. Regression tests
include the CLI segment chain (currently a strict xfail citing that line).

### Facts corrected this round

- `/staging/jsstrobel` **does not exist** (earlier "empty, 2.7 PB free" read a
  failed `ls` as zero). Outputs stay in `/home` (4.9 GB headroom) until the
  user requests staging from CHTC.
- 138 held jobs in the queue: 36 reference an `image.sif` that no longer exists
  (dead); the rest are Paper I/II campaigns — user's call, not touched.

## Adversarial verify round — 2026-09-03 (9 refuters over `9a576e4a`)

| Lane | correctness | reproduce | scientific surface |
|---|---|---|---|
| graph-span `af222b69` | holds (claim fix required) | holds — 54 passed; 19q oracle builds in 0.5–0.8 s | holds — default H2O sha `34683ac8…`, route sha `6ed3f6a4…` identical to pristine; locked contract untouched |
| ledger port `34c87900` | holds — every slice→window bound verified; ledger append-only | holds — 96 passed, 6 skipped; parity re-run: ledgers byte-identical | holds — fresh 1-round H2O parity: `estimator_call_ledger.json` bytes identical (2,224,160 B), energy −74.94152757586416 equal |
| wall-clock `929f1b47` | holds on its own code | holds on its own code | **refuted at the merged head** |

### Must-fix items the refuters found

1. **`34c87900` broke every accepted-state resume** (HH included): `adapt_pipeline.py:29943`
   references `occurrences`, a binding the port deleted from
   `_hydrate_default_no_prune_numerical_session` → `NameError`. Four resume
   tests pass at `6648ed5b` and fail at `9a576e4a`. The ledger lenses missed
   it because they did not run resume tests. Fix: windowed
   `_default_no_prune_ledger_occurrence_rows(restored_ledger, 0, restored_occurrence_count)`
   — already present in the resume builder's worktree.
2. **Singleton arm cannot finish.** `sector_complete_pauli_block_v1` exits 1
   after its first accepted round at `engine.py:3319`: the staged-child
   lineage branch opens only for `candidate_representation == single_pauli_word_v1`,
   but the H2O sector-complete mode resolves to `macro_generator_v1` while its
   accepted rows are staged children `h2o_sector_block::<hash>`. Sibling sites
   `:3516`, `:3610`, `:3629`. Pre-existing on main, masked there by the earlier
   exit-1 that `929f1b47` fixed. The wall-clock lane's "H2O completes" claim
   holds for the macro arm only until this is fixed.
3. **The 19q embedding is a choice, not a derivation.** The six historical
   entries admit 430 tree-valued two-arm continuations; choice points at
   logical index 12 (8 vs 17), 13 (20 vs 36), 15 (dead end). Named
   alternatives and their cost on the tracked 19q register (boson modes at
   logical 12–13, 14–16, 17–18):

   | Embedding tail after the 12q prefix | Boson-block Steiner edges | Σ pairwise dist | Diameter |
   |---|---|---|---|
   | lane (current) `…8,20,9,24,10,25,11` | (10, 8, 8) | 862 | 13 |
   | B `…17,20,27,36,26,41,25` | (8, 7, 6) | 784 | 10 |
   | F row-0 fill `…8,9,10,11,12,13,14` | **(1, 2, 1)** | 1008 | — |

   The choice changes every graph-span cost reported for H2O. **User decision.**
4. H2O `--resume-checkpoint` refused at `_resume.py:1455` — the active build.

### Package fix-up `216aec84`

`next_segment.py` now keys on the real stop reasons (`wall_clock_budget_v1`
continue; `exact_ed_target_reached` / `maximum_controller_rounds` done);
preflight 21/21 on the merged HEAD. Macro arm measured end-to-end under the
runner env with compact layout: `--max-wall-seconds 0` → exit 0, 368 s,
394 MB, `result.json` with `primary_reason == wall_clock_budget_v1`, seg01
renders with `--resume-checkpoint` and matching sidecar digests.

## Round 3 — 2026-09-03 (Opus 5 builders; merged at `315b96e2`)

| Commit | Lane | Evidence |
|---|---|---|
| `73854cbc` | H2O admitted to canonical accepted-state resume: named `paper_iv_h2o_linear_fd_application` branch (family + problem key + route family/profile prefix + canonical interface + application lane + parent contract sha), fixture-sha **attestation** written into the checkpoint for the H2O family only (15 fields, no path strings), 14 required route settings read from the canonical H2O v3 contract; HH/pure-Hubbard branches byte-identical. Also the exact `NameError occurrences` fix. Lane contract `molecular-vibronic/AGENTS.md` rewritten. | 14 passed + 1 xfail (1113 s opt-in); macro arm seg1 depth 1 `E=−74.955593524326` → seg2 from a **relocated** fixture path depth 2 `E=−74.959928311698`; also under compact layout |
| `eecf2b8f` | Sector-block lineage: parallel fail-closed validation branch keyed on adapter capability + row schema (relaxing the `single_pauli_word_v1` gate could not work — the guarded body demands one term + runtime-split metadata); sibling sites `:3516/:3610/:3629` untouched, Paper-I path byte-identical | 10 passed; opt-in sector arm 2 rounds → exit 0 + `result.json`, 294 s, 534 MB; 190 passed / 9 pre-existing across five suites |
| `3bcf490f` | Embedding claim made honest (chosen, not derived; alternatives B/C/D/F with spans in the code comment), two coupling-map property tests, `(10,8,8)` boson-block spans pinned against the tracked fixture, `validate_cost_mode_register_size` before any file is written (0.42 s fail on the 15q fixture, nothing created) | 65 passed; default sha `34683ac8…` unchanged |
| `315b96e2` | merge of the three | fast baseline **237 passed, 21 skipped, 1 pre-existing failure**; package rebuilt, preflight **21/21** |

Sector-arm end-to-end resume test un-xfailed after the merge and **confirmed**:
opt-in chain run on `315b96e2` (`PAPER_IV_H2O_CLI_RUNS=1`, both files,
`-k "sector_arm or segment_chain or macro_arm or wall_budget_stops"`) →
**6 passed** in 27 min 45 s, peak RSS 867 MB under `ram_guard`. Both arms now
stop under `--max-wall-seconds`, exit 0 with `result.json`, and resume from
their own `current.json` + sidecar into a fresh output directory.

### Still gated on the user

1. Embedding choice — lane (10,8,8) / B (8,7,6) / F (1,2,1). Recommendation: F.
2. Install the opus-only subagent hook (`.claude/hooks/require-opus-subagents.py`, self-tested).
3. Remove the 36 held jobs whose `image.sif` no longer exists.

### Stale, not regressed

`agent_guidance/static-adapt/failure-atlas.md` was generated 2026-08-18 at
`ce2adf79` by `pipelines/shell/failure_atlas.py`; it is hundreds of commits
stale independent of this work.

## Production fixture from the vibronic builder — 2026-11-03

Two decisions by Jake on 2026-11-02 change the Paper IV model definition
(scientific record: `work/doron-chemistry-brief-20261102.md` in the main
checkout, uncommitted; measurement scripts `work/vibronic_curvature_20261102/`):

1. **Oscillators** omega_nu^2 = K_nu^PES + chi_nu: the frozen-core FCI surface
   curvature along the mode plus the model's own correlated response to the
   assembled coupling, so the exactly solved model carries the FCI curvature.
   The RHF Hessian used so far holds the orbital relaxation that the coupling
   regenerates (double counting: the old model's adiabatic symmetric-stretch
   curvature is about a third of the FCI surface's, 2237 against 3998 cm^-1,
   and its ground state holds 2.3 quanta on that mode at cutoffs (4,8,4),
   hence the n_max = 7 of the production run).
2. **Modes** = eigenvectors of K^PES + chi as full matrices in the RHF
   normal-mode basis (model-consistent modes), so the full FCI Hessian is
   reproduced rather than its diagonal; bend and symmetric stretch (A1) mix by
   2.6 degrees. Displaced orbitals are aligned block by block so the frozen
   core stays canonical: the full-set Procrustes rotation of the old generator
   added about 1e-4 Ha/Q^2 of spurious curvature and moved the couplings by
   0.5-5 %.

Implemented in the molecule-agnostic builder
`src/quantum/chemistry/molecular_vibronic.py` (branch
`claude/molecular-vibronic-builder-20261102`, commits `b5426f27` / `0ad967dc`
/ `72a0668b`; cherry-picked onto this branch as `0513703f` / `9951ab62` /
`d5dac21d`; the legacy model is reproduced with `harmonic_curvature=
"rhf_hessian", alignment="full"` to 2e-7 Ha).

**Producer.** `src/quantum/chemistry/paper_iv_h2o_fixture_from_builder.py`
builds the lane's `h2o_linear_fd_backend_record_v1` from the builder's model
and hands it to the lane's own generator
(`generate_h2o_linear_fd_fixture.build_h2o_linear_fd_fixture_from_record`).
Schema decision: convert to the record rather than teach the lane the builder
schema, so derivative validation, operator encoding, exact reference,
evidence gate and sha attestation are unchanged. The record's modes are the
model modes, labelled bend / symmetric_stretch / antisymmetric_stretch by
internal-coordinate character (the lane's policy) and kept in that order, so
positional conventions survive; the displaced tensors are real SCFs along the
model modes; the generator's central difference reproduces the builder's
rotated couplings to 2e-6 Ha/Q. Drift validation: production tier at relative
tolerance 1e-5, the setting the 2026-08-26 fixture was validated with.

**Fixture.** `test_support/paper_iv_h2o/h2o_builder_model_consistent_work333_ref444_20261103_fixture.json`
(record and summary beside it):

| quantity | new fixture (3,3,3) | old fixture (3,7,3) |
|---|---|---|
| sha256 of the bytes (the attested identity) | see the committed summary file (`<tag>_summary.json`, `fixture_sha256`), pinned in `test_checked_in_production_fixture_matches_its_summary` | `190387449b556904c73fd424635e494eb92af2c13e8b0e0d71234a4bc391256e` (re-encoded from source fixture `fe3c041e…`) |
| bytes | 15,673,312 (sha256 `de32cd4d1e29b8dcc07e94ad132ee43893fe03fd36cf4538eb6dadf6d5e194df`) | 28,025,185 |
| qubits / sector dimension | 18 / 14,400 | 19 / 28,800 |
| modes bend / sym / asym (cm^-1) | 2130.79 / 5314.42 / 5061.82 | 2170.79 / 4138.69 / 4389.24 |
| exact ground energy (Ha) | −74.99991178 | −75.01446351 |
| reference cutoffs, energy, delta | (4,4,4), −75.00006177, 1.5e-4 Ha, passed | (4,8,4), −75.01513494, 6.7e-4 Ha, passed |
| builder's own solution of the same model | −74.99991177 (differs by 1.1e-8 Ha) | — |
| drift scaled residuals bend / sym / asym (rel 1e-5) | 0.17 / 0.06 / 0.11 | 0.39 / 0.06 / 0.12 |

Evidence gate `validate_paper_iv_h2o_linear_fd_evidence_fixture`: passed;
`production_validated`; the registry's cached loader
(`load_cached_production_vibronic_h2o_linear_fd_fixture`) loads it. Tests:
`test/test_paper_iv_h2o_fixture_from_builder.py` (record structure; small
fixture end to end, 93 s; checked-in fixture against its summary; the
production run itself is opt-in, `PAPER_IV_FIXTURE_FROM_BUILDER_FULL=1`).

**What a restart on the new model needs (nothing submitted or stopped):**
(a) the request must name the new fixture path and carry `n_ph_max = 3` (the
registry checks it against the fixture's largest cutoff; the old request used
7; `h2o_application` reads it from the fixture, so the package runner is
unchanged); (b) a package for the new fixture — **built 2026-11-03, see
below**; the route contract's `paper_iv_h2o_fixture_source_lock_v1` is a
name, not a hash, so no contract change for the fixture itself, but the
attestation makes old checkpoints non-resumable on the new fixture by design;
(c) Jake's call on the running seg13 (cluster 12396953.0, old model); (d) the
manuscript has to state the new model: K^PES + chi oscillators,
model-consistent modes, 2.6 degrees of A1 mixing, 18 qubits, cutoffs 3.

### CHTC package for the decided model — 2026-11-03 (user: "Yes, build it")

`chtc/paper_iv_h2o_two_arm_model_consistent_20261103/`: the
`paper_iv_h2o_two_arm_20260903` package (same segment discipline, arms,
runner, chaining, monitor) with the campaign name, fixture, qubit count,
memory request and a fixture-identity guard changed. Non-submitting; nothing
was sent to ap2001.

| | value |
|---|---|
| fixture packaged | `h2o_builder_model_consistent_work333_ref444_20261103_fixture.json`, sha256 `de32cd4d…94df`, 15,673,312 bytes |
| guard (`build_package.py`, re-verified by `runtime_preflight.py`) | bytes, sha256, 18 qubits, `n_ph_max_by_mode == [3,3,3]`, lane mode order, `production_validated`, `model_definition.decided_model` with `pes_consistent` / `block_diagonal` / `model_consistent`; the old fixture is refused (`n_total_qubits 19 != 18`) |
| source archive | `git archive` of HEAD `df206077` (`source.tgz` sha256 `3039eaa3…`), engine and registry unchanged from the live refit branch `d17d4046` |
| `STATIC_ADAPT_PAULI_SIGN_CACHE_MAX_NQ` | 18 (was 19) |
| `request_memory` | 16GB default, `--request-memory` build option; derivation: the 19q (3,7,3) campaign's measured true peak 14,792 MB at round 25 (raised to 22,528 MB) scaled by the sector ratio 14,400/28,800 with a factor-of-two margin; no peak measured for this model yet |
| `--max-wall-seconds` | 129,600 s default (unchanged; rounds are cheaper at half the sector) |
| graph-span cost mode | needs an 18-qubit embedding decision; the 19-qubit spans (10,8,8)/(8,7,6)/(1,2,1) were for 2+3+2 boson qubits, now 2+2+2 |
| `runtime_preflight.py` | **pass, 23/23** (21 inherited + `fixture_identity_verified`, `manifest_fixture_identity_matches`); archived `h2o_application --help` lists every runner flag and both arms |
| tests | `test/test_paper_iv_h2o_two_arm_model_consistent_package.py`: 4 pass (constants, identity guard accepts only the decided fixture and rejects a same-length edit, submit rendering with the campaign/memory, built-manifest identity) |

Before any submission (user): decide the fate of seg13 of the old model,
confirm `image.sif` and quota headroom on ap2001, rsync the directory, point
the monitor/watcher at the new batch names.
