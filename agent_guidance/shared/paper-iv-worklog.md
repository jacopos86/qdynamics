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
