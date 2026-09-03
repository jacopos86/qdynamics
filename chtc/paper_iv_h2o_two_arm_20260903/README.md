# paper_iv_h2o_two_arm_20260903 — campaign contract

Non-submitting CHTC package for the Paper-IV H2O static construction
(19 qubits, sector dimension 28,800, work cutoffs (3,7,3)), run as **bounded
segments that exit normally** so every segment's results transfer. Nothing in
this directory submits, opens SSH, or changes remote state.

Handoff authority: `agent_guidance/shared/paper-iv-h2o-chtc-handoff-20260903.md`
(read §0, §3, §4, §5). Precedent packages studied:
`chtc/paper_iv_h2o_sr_paper_i_noprune_nobeam_derivative_resolved_d50_20260724/`
and the lost run `~/chtc_rescue/home/chtc/d50pkg/{run_d50.sh,submit.sub}`.

## Why segments

Cluster 9715945 ran 70.0 h and returned nothing: CHTC's 72 h limit is a
`SYSTEM_PERIODIC_HOLD`, and a hold never stages out (`StageOutStart`,
`SpooledOutputFiles` undefined; `condor_transfer_data` found no `results/`).
`ON_EXIT_OR_EVICT` did not help; `+MaxRuntime` enforces nothing by itself;
there is no SIGTERM handler in the engine, so `kill_sig_timeout` buys nothing.
The only reliable protection is that **the Python process exits normally before
72 h**. Everything here follows from that.

## The two arms

| Arm | `--candidate-representation` | JobBatchName |
|---|---|---|
| `macro` | `macro_generator_v1` | `paper_iv_h2o_two_arm_20260903_macro` |
| `singleton` | `sector_complete_pauli_block_v1` | `paper_iv_h2o_two_arm_20260903_singleton` |

`single_pauli_word_v1` is **not** an arm: `h2o_application._candidate_adapter`
refuses it for H2O (`pipelines/static_adapt/ra_adapt/h2o_application.py:76-80`)
and `run_segment.sh` refuses it before launching.

Both arms, every segment: `--insertion append_only`,
`--checkpoint-history-tail 1`, `--maximum-controller-rounds 50` (the campaign
horizon, not a per-segment count), estimator ledger **enabled** (resume
validates its digest-matched sidecar, `sr_snake/_resume.py:4047-4073`), and
`--cost-mode <proxy|marrakesh_graph_span_v1>` chosen per arm at build time
(`build_package.py --macro-cost-mode/--singleton-cost-mode`, default `proxy`).
Graph-span at 19 qubits is blocked on a user decision (handoff §0.1); the
package parameterizes it but does not choose it.

## Segment discipline (the numbers)

| Boundary | Seconds | Where |
|---|---|---|
| Pool wall (hold, no stage-out) | 259,200 (72 h) | CHTC policy |
| `periodic_hold` (belt and braces) | 172,800 (48 h) | every submit file |
| `--max-wall-seconds` (the real stop) | 129,600 (36 h, default) | `run_segment.sh` argument |

The engine checks the wall budget **between controller rounds**, after the
completed round has been checkpointed, and returns normally exactly the way
a `maximum_controller_rounds` stop does: `current.json` + its sidecars,
`result.json` with `run.stop.primary_reason == "wall_clock_budget_v1"`
(`sr_snake/_controller.py` `WALL_CLOCK_BUDGET_STOP_REASON`), `summary.json`
with `stop_fired_reasons` / `wall_clock_elapsed_seconds`, Python exit 0. A
protocol stop that fires in the same round keeps precedence (`fired_reasons[0]`),
so a wall-stopped segment is by construction one with no scientific stop. The
12 h between the in-process stop and the `periodic_hold` covers the last
round (the most expensive of a segment: append-only ADAPT grows per round),
the final checkpoint write, the tarball, and output transfer. Nobody has
measured per-round seconds at 19 qubits yet (handoff increment 2); once that
exists, rebuild with `--max-wall-seconds` sized so that
`budget + max_round_seconds + 1 h < 172,800`.

The `periodic_hold` at 48 h does **not** save the segment (a hold never
stages out). It saves 24 h of pool time and makes the failure visible while a
human can act. `max_retries = 0`: a retry would restart from round zero.

Requests: `request_cpus = 4`, `request_memory = 8GB` (measured 4,151 MB RSS
peak, +34 MB/day), `request_disk = 10GB` (measured 293 MB). All BLAS/OpenMP
thread pins are 1, as in the lost run.

## What transfers, when

`when_to_transfer_output = ON_EXIT`. Only a normal exit of `run_segment.sh`
transfers anything; the runner propagates the Python exit code, and a nonzero
code still transfers (`on_exit_hold` is deliberately absent).

Results mode is a build-time switch (`build_package.py --results-mode`):

| Mode | `transfer_output_files` | Lands at |
|---|---|---|
| `home` (default) | `results` | `/home/jsstrobel/chtc/paper_iv_h2o_two_arm_20260903/<arm>/segNN/results/` |
| `staging` | `transfer/<arm>_segNN_results.tar.gz` + `transfer_output_remaps` | `/staging/jsstrobel/paper_iv_h2o_two_arm_20260903/<arm>/segNN/<arm>_segNN_results.tar.gz` |

**Why `home` is the default:** the handoff's 2026-09-03 correction verified
with `ls -ld` that `/staging/jsstrobel` does **not** exist (an earlier read
counted a failed `ls` as "empty"). A remap into a missing directory fails the
output transfer — the same loss as a hold. Per-user `/staging` is provisioned
by CHTC facilitators on request, which only the user can do. Once it exists,
rebuild with `--results-mode staging`; nothing else changes.

**Why a tarball + `transfer_output_remaps` in staging mode, not a directory
remap:** the only remap this pool has been shown to honor is a single file
remapped to `/staging` (`chtc/paper_i_hh_b3_measured_residual_20260821/b3mr_18cell_chtc/submit_fresh.sub:13`,
`chtc/paper_i_ra_adapt_repair_20260727/paper_i_ra_allphase_adaptive_forced_k50_four_arm_20260817_v2_chtc/submit_home.sub:13`;
the rescue fetch recovered ~65–78 GB from `/staging` written that way). No
package in `chtc/` has ever remapped a directory, so `run_segment.sh` tars
`results/` first.

`/home` headroom is 4.9 GB (35,719 MB used of 40,960 MB soft). Each segment
adds `results/` (order 10–100 MB with `--checkpoint-history-tail 1` and the
compact layout) plus logs; the package itself is ~34 MB. The monitor alerts
at 38,000 MB. **Retrieve, verify, then ask the user before deleting anything
remote.**

## What is user-gated

Every one of these is an external-state change and is never performed by
anything in this directory:

1. `ssh -MNf ap2001` (one Duo push; the ControlMaster socket is reused).
2. `rsync -av chtc/paper_iv_h2o_two_arm_20260903/ ap2001:chtc/paper_iv_h2o_two_arm_20260903/`
   (after `build_package.py` and a passing `runtime_preflight.py`).
3. Confirming `image.sif` still exists at
   `/home/jsstrobel/paper_i_hh_b3_mr_measured_costexp_20260824/image.sif`
   (36 held jobs in the queue died on a vanished image) and that `quota -s`
   shows headroom.
4. `condor_submit` of each `submit.sub` (seg00: `submit_macro.sub`,
   `submit_singleton.sub`; later: `<arm>/segNN/submit.sub` from
   `next_segment.py`), then recording the ClusterId in `segment.json`.
5. Deleting retrieved segments from `/home`, releasing or removing any job
   (`condor_release` restarts from round zero; `condor_rm` on a held job
   destroys any spooled sandbox), requesting `/staging`, and triaging the 138
   held jobs (only one is Paper IV).
6. Choosing `marrakesh_graph_span_v1` (needs a 19-qubit FakeMarrakesh
   embedding, handoff §0.1) and adopting or demoting any result as evidence.

## Files

| File | Role |
|---|---|
| `build_package.py` | `git archive` of the worktree **HEAD** (sha recorded; uncommitted edits are not shipped) → `package/source.tgz`; copies the fixture; writes `package/input_manifest.json` (+ `.sha256`), `package/build_record.json`, `submit_macro.sub`, `submit_singleton.sub`, `<arm>/seg00/{submit.sub,segment.json,logs/}`. Self-checks every submit file as `key = value` lines. |
| `run_segment.sh` | Execute-node runner. Verifies input digests with `sha256sum -c`, unpacks the source, moves any resume files into `resume/` (never inside `--output-dir`), runs `h2o_application` under Apptainer with the lost run's env pins plus `STATIC_ADAPT_CHECKPOINT_JSON_LAYOUT=compact`, writes `results/segment_receipt.json`, tars in staging mode, exits with Python's code. |
| `next_segment.py` | From a retrieved `results/`: reads `result.json` `run.stop.primary_reason` / `fired_reasons` (cross-checked against `summary.json`) and continues **iff** `primary_reason == "wall_clock_budget_v1"`; reports the arm finished when `exact_ed_target_reached` or `maximum_controller_rounds` (the 50-round horizon) fired, `summary.json` says `chemical_accuracy_reached`, or the checkpoint energy is within tolerance of `protocol.json`'s exact target; refuses anything else (no `result.json`, an unknown reason) with the reason printed (`--resume-after-abnormal-exit` is the explicit override for a crashed/held segment, never for a scientific stop); verifies `current.json` (`iteration_done`, `complete: false`, both estimator-ledger pointers agree, sidecar sha256 matches, sidecar schema is the checkpoint-sidecar schema, every other content-addressed sidecar likewise); stages them into `<arm>/seg{N+1}/input/` with `resume_manifest.sha256`; renders `submit.sub` with `--resume-checkpoint` and a fresh output dir. `--verify-only` for retrieval checks. |
| `monitor.sh` | Hourly read-only watch (handoff §5.2): job state, run time vs budget (warn at 40 h), hold alerts, `quota -s` alert above 38,000 MB, 12-hourly rsync mirror to `~/paper_iv_h2o_results/` followed by `next_segment.py --verify-only` on every retrieved `current.json`. Exit 78 when `ssh -O check ap2001` fails. |
| `runtime_preflight.py` | Local, no SSH: `--help` of the **archived** `h2o_application` lists every flag the runner passes and both arms; fixture/runner/source digests match the manifest; the archive imports; submit files parse; `py_compile` and `bash -n`. Exit 2 when blocked. |

Segment naming: `<arm>/seg00`, `seg01`, … Each segment directory is the
job's `initialdir` on ap2001 and holds `submit.sub`, `segment.json`, `logs/`,
`input/` (segments > 00) and, after transfer, `results/`.

## Workflow

```bash
PY=/Users/lordskyler/Holstein_test/.venv/bin/python
cd /Users/lordskyler/Holstein_test.worktrees/paper-iv-h2o-campaign
$PY chtc/paper_iv_h2o_two_arm_20260903/build_package.py            # writes package/, submit_*.sub
$PY chtc/paper_iv_h2o_two_arm_20260903/runtime_preflight.py        # must print "status": "pass"
# --- user: ssh -MNf ap2001; rsync the directory; condor_submit submit_macro.sub / submit_singleton.sub
bash chtc/paper_iv_h2o_two_arm_20260903/monitor.sh <cluster>,<cluster>   # hourly (cron / launchd / /loop)
# --- after a segment returns (monitor mirrors it under ~/paper_iv_h2o_results/mirror/home/<arm>/segNN/results/)
$PY chtc/paper_iv_h2o_two_arm_20260903/next_segment.py --previous-results ~/paper_iv_h2o_results/mirror/home/macro/seg00/results
# --- user: rsync macro/seg01/ to ap2001 and condor_submit it
```

`runtime_preflight.py` passes on the campaign branch at `9a576e4a`
(`--cost-mode` from `af222b69`, `--max-wall-seconds` from `929f1b47`,
`--resume-checkpoint` from `9a43bfe2`); the archive is HEAD, so any later
engine change must be committed before rebuilding. Besides the flag/digest
checks it verifies, in the **archived** source, the stop-reason strings and
ledger-sidecar schemas `next_segment.py` hard-codes, the H2O finalization
skip in `runtime.py`, and that `run_segment.sh` stages resume files in
`resume/` rather than inside `--output-dir` (`h2o_application` refuses a
checkpoint inside its output directory).

## Blockers measured on the integration branch (6648ed5b), 2026-09-03 — status at 9a576e4a

Both were found by running `h2o_application` locally on the July fixture
(`chtc/paper_iv_h2o_sr_paper_i_noprune_nobeam_derivative_resolved_d50_20260724/input/h2o_fixture.json`,
1 controller round, ~3 min, each arm). Blocker 2 is **fixed** at `9a576e4a`
(`929f1b47`, `runtime.py` skips the Paper-I canonical summary for the
`molecular_vibronic_h2o_linear_fd` family; verified locally for the macro arm
with `--max-wall-seconds 0`: exit 0 in 368 s / 394 MB RSS, `result.json` +
`summary.json` written, `stop_reason == wall_clock_budget_v1`; the singleton
arm now fails earlier, see 3). Blocker 1 **still stands**
(`test/test_ra_adapt_h2o_application.py::test_h2o_cli_segment_chain_resumes_from_the_wall_clock_stop`
is a strict xfail on it).

1. **`--resume-checkpoint` cannot resume an H2O checkpoint.**
   `pipelines/static_adapt/sr_snake/_resume.py:1456-1460`
   (`_validate_problem_binding`) raises
   `CanonicalResumeError: Canonical accepted-state resume is restricted to
   canonical L=2 Hubbard--Holstein or the exact named L=2 pure-Hubbard
   controller-noise application.` before any round runs. The staged
   `current.json` + sidecars produced by `next_segment.py` pass every
   digest/pointer check this package performs; the engine refuses the problem
   binding itself. Until H2O is admitted there (a scientific-surface change,
   not a package change), **segment 01 of either arm cannot start**, and a
   campaign of bounded segments reduces to one segment per arm.
2. **(Fixed at 9a576e4a.) Finalization crashed for both arms after the
   horizon was reached.** At `6648ed5b`, `runtime.py` routed `append_only`
   RA-ADAPT results through `summarize_paper_i_run`, which raised
   `canonical Paper-I summary route identity disagrees with the typed
   canonical route authority.` (`paper_i_run_summary.py:1630`), so
   `current.json` was written but `result.json`/`summary.json` were not and
   the process exited 1. `929f1b47` makes the H2O family skip that summary;
   `runtime_preflight.py` checks the archived `runtime.py` carries the skip.

3. **(New at 9a576e4a, unmasked by the fix to 2.) The singleton arm
   crashes at finalization after its first accepted round.** With
   `--candidate-representation sector_complete_pauli_block_v1`, both
   `--max-wall-seconds 0` and `--maximum-controller-rounds 1` (no wall flag)
   exit 1 from `pipelines/static_adapt/ra_adapt/engine.py:3319`
   (`_accepted_candidate_lineage_receipts`):
   `RuntimeError: Accepted RA candidate is absent from the authenticated
   executable inventory.` The accepted row is the staged sector block
   `h2o_sector_block::<sha16>` / `generator_id h2o-sector-block:<sha16>`,
   which is not an entry of `adapter.executable_pool(problem)`; the
   staged-child branch of that function opens only when the protocol's
   `candidate_representation == single_pauli_word_v1`, and the H2O
   sector-complete mode resolves to `macro_generator_v1`
   (`launch_manifest.json`: `candidate_mode` vs `candidate_representation`).
   `current.json` + sidecars are written (depth 1, `complete: false`);
   `result.json`/`summary.json` are not. On pristine main `c81247de` the
   same run dies earlier, in blocker 2's `summarize_paper_i_run`
   (`runtime.py:1969`), so this site was never reached there: the engine
   crash is pre-existing but masked, not introduced by the merged lanes.
   The macro arm is unaffected
   (exit 0, `result.json`, `stop_reason == wall_clock_budget_v1`). Until
   this is fixed in the engine (a scientific-surface change, not a package
   change), **the singleton arm returns a checkpoint but never a result**,
   and `next_segment.py` classifies it `abnormal_exit`.

None of these was caused by this package; 1 and 3 are not fixed here.

## Retrieval is only trustworthy once verified

A returned `results/` proves nothing until `next_segment.py --verify-only`
passes on it (exit 0: resumable; exit 3: checkpoint verified but the arm is
finished or the stop was not `wall_clock_budget_v1`; anything else: the
retrieval failed verification). A normally-exited segment has `result.json`
and `summary.json`; `current.json` must be an `iteration_done` checkpoint whose
estimator-ledger pointer (`checkpoint.estimator_call_ledger_checkpoint` ==
`adapt_vqe.estimator_call_ledger_checkpoint`) names a sibling
`current.estimator_call_ledger_checkpoint.<sha16>.json` whose SHA-256 matches;
`current.verified_singleton_resume.<sha16>.json` is verified the same way.
`segment_receipt.json` records the Python exit code, elapsed seconds, image
sha256, ClusterId, and the digest of every results file.

Report numbers, not verdicts: completed rounds and stop reason per segment,
energy against `E_ref = −75.015134943251 Ha`, whether the 1.6 mHa stop fired.
Evidence adoption is the user's decision.
