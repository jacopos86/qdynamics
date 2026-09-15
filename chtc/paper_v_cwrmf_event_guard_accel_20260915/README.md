# Paper V CWRMF event-guard acceleration lane (2026-09-15)

Bounded-segment CHTC lane for the decisive cone result of the plan of record
(`agent_guidance/paper-v/paper-v-state-and-plan-20260914.md`, section 4.2 step
6): the bounded-depth event-triggered Gram guard on the 384-coordinate
carried-witness radial moment flow, balanced numerical profile, event Gram
hard floor `2e-4`, bounded step halving at depth 2, over the full driven
interval `t = 0` to `4` at `h = 0.0025`, scored against the exact cutoff-16
reference the driver regenerates in process.

Supersedes nothing.  The 2026-08-18 lane
`chtc/paper_v_cwrmf_event_guard_20260818/` remains as written; it is a single
unbounded `t = 0.5` job at 1 CPU and 8 GB and was never uploaded.  This lane
differs in four ways: bounded segments that exit normally and resume from
`checkpoint.npz`, 8 CPUs for the named `max_threads_8` solver setting set,
12 GB for the recorded `6.05 GB` peak, and receipt-named stage-out.  A
segment is bounded twice: by its `0.5` unit of propagated time and by the
driver's declared wall budget of `138240 s`.

## Why segments, and why `0.5` time units each

The local completion of this exact guard (balanced, `--event-gram-hard-floor
2e-4`, `--maximum-step-subdivisions 2`) took `14,722 s` for `0.38` time units
of guarded propagation -- `t = 0.1175` to `0.5`; the prefix below the trigger
is nearly free -- with a `6.05 GB` peak.  That is `10.8 h` per unit of
propagated time on the replacement M4 Pro.

| segment length | expected wall at the local rate | at a 2x execute-node derate | at 4x |
|---:|---:|---:|---:|
| `0.5` | `5.4 h` | `10.8 h` | `21.5 h` |

`+MaxRuntime` is `48 h` and the pool holds any job at `72 h` with no
stage-out, so `0.5` per segment leaves the whole derate range inside the
bound.  Eight segments cover `t = 0` to `4`.

The rate beyond `t = 0.5` is **unmeasured**: no guarded CWRMF run has ever
passed `t = 0.5`, and the guard fires more often deeper into the pulse.  Treat
`0.5` as an upper bound on segment length: if segment 02 returns a wall time
above about `12 h`, split the remaining segments before submitting them.

## The second bound: wall time

Segment length bounds the propagated interval, not the wall clock, and the
rate beyond `t = 0.5` is unmeasured, so segment length alone cannot keep a
job inside `+MaxRuntime`.  The inner runner therefore passes
`--maximum-wall-seconds 138240` to the driver, which is `0.8 x 172800`.  At
the first accepted SSPRK2 step that ends past that budget the rollout stops,
`checkpoint.npz` holds that step, the trajectory arrays and `summary.json` are
written for the steps taken with `stop_reason = wall_budget`, and the process
exits 0, so the job ends normally and stages out.  The remaining fifth of the
runtime is the headroom the offline exact-reference scoring, the artifact
writing and the archiving need after the rollout ends.

A segment that stops this way has **not** reached its declared `final_time`.
Submit that same segment's `final_time` again, from the checkpoint it left,
with a new `run_id`; the driver reads the resume time from the checkpoint, so
nothing else in the submit file changes.  `stop_reason` and `last_time` in
`receipt.json` are what say which case a returned segment is.

`CWRMF_MAXIMUM_WALL_SECONDS` overrides the budget for one segment.  A shorter
budget is the right response to a segment that returns `wall_budget` with very
little propagated: it makes the next attempt's stop-and-stage cycle shorter,
not slower.

## Argument contract

```
run_id  final_time  time_step  event_gram_hard_floor  initial_state_file
```

`initial_state_file` is the previous segment's `checkpoint.npz` relative to
the job working directory, or the literal `none` for the first segment.  The
driver reads the resume time from the checkpoint, so `final_time` is always
absolute.  The eight submit files already carry the contract; do not hand-edit
a segment's profile, cutoff, coupling, drive, hard floor or step.

## Job shape

Single vanilla job per segment.  8 CPUs (the named `max_threads_8` Clarabel
setting set, which is why this lane is not the 2026-08-18 lane's 1 CPU),
12 GB memory against the recorded `6.05 GB` peak, 30 GB disk (image, scratch
venv, trajectory and checkpoint), `+MaxRuntime = 172800`, against which the
inner runner passes the driver a `138240 s` wall budget.

The import chain is self-contained: `paper_5/src` plus numpy, scipy, cvxpy and
clarabel.  The inner runner bootstraps a scratch venv with pinned versions
only when the time-dynamics image lacks the conic stack.

## Named options in this lane

Every option below is off or at its recorded default unless the environment
sets it; nothing in this lane changes a default of the model.

| environment variable | default here | meaning |
|---|---|---|
| `CWRMF_SOLVER_SETTING_PROFILE` | `max_threads_8` | the named Clarabel setting set; it is qualified against the balanced full-Gram acceptance test, not the strict one, and it is not bitwise |
| `CWRMF_BUNDLE_SEED_POLICY` | unset, i.e. `monotone_hint` | the starting working set of the adaptive bundle tier |
| `CWRMF_BUNDLE_SEED_RANK_MARGIN` | unset | read only when the policy is set |
| `CWRMF_MAXIMUM_WALL_SECONDS` | `138240`, i.e. `0.8 x +MaxRuntime` | the driver's declared wall-clock budget; the rollout stops at the first accepted step past it, exits 0 and records `stop_reason = wall_budget` |

### Lane acceleration decision

`--bundle-seed-policy carried_accepted_basis --bundle-seed-rank-margin 2` is
**not** the default of any submit file here, and the reason is measured, not
cautionary.  Recorded in
`agent_guidance/paper-v/overnight/bundle-rank-seed-20260915.md` section (c)
part two: over `t = 0` to `0.05` at `h = 0.0025` under the **balanced**
profile with the always-on guard, that pair costs `879.775 s` of Clarabel
solve tier against the balanced default's `257.439 s`, a `3.42x` regression,
with the maximum readable-rate residual at `2.8753e-03` against `2.9030e-05`
and a peak resident set of `3876 MiB`.  The balanced `psd_inflation = 1e-8`
lifts the predictor spectrum above the strict entry threshold, so the balanced
tier does not walk its working set and carrying that set only enlarges it.
The `2.04x` the pair earns is a **strict**-profile result on `t = 0` to
`0.01`.

`max_threads_8` alone is the acceleration this lane carries: `1.49x` on the
strict `t = 0` to `0.01` Clarabel tier, measured, and qualified against the
balanced acceptance test, which is the test this lane runs.

To try the carried policy on a segment anyway, set the two variables in the
submit file's `environment` line for that one segment and give it a new
`run_id`; do not change the defaults here.

## Upload from local macOS (repo root of this worktree)

The bundle is built for you at
`output/local_runs/paper_v_cwrmf_event_guard_accel_20260915/`; its SHA-256 and
byte count are in `bundle.sha256` and `bundle.bytes` beside it.  The Apptainer
image is **not** in the bundle (206 MB, not a repository file).

```bash
ssh -MNf ap2001                      # the user runs this once; it needs Duo

ssh ap2001 "mkdir -p ~/Holstein_paper_v_cwrmf_accel_20260915/chtc/time_dynamics_optuna"

scp /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_cwrmf_event_guard_accel_20260915/paper_v_cwrmf_event_guard_accel_20260915.tgz \
  ap2001:~/Holstein_paper_v_cwrmf_accel_20260915/

scp /Users/lordskyler/Holstein_test/chtc/time_dynamics_optuna/image.sif \
  ap2001:~/Holstein_paper_v_cwrmf_accel_20260915/chtc/time_dynamics_optuna/image.sif
```

Skip the image copy if a Paper II root on the AP already holds the same
`image.sif`; hard-link it into this root instead of copying 206 MB again.

## Remote extract, validate, submit (on ap2001)

```bash
ssh ap2001 'cd ~/Holstein_paper_v_cwrmf_accel_20260915 \
  && sha256sum paper_v_cwrmf_event_guard_accel_20260915.tgz \
  && tar -xzf paper_v_cwrmf_event_guard_accel_20260915.tgz \
  && chmod +x chtc/paper_v_cwrmf_event_guard_accel_20260915/*.sh \
  && mkdir -p logs raw_outputs archives_by_receipt'

ssh ap2001 'cd ~/Holstein_paper_v_cwrmf_accel_20260915 \
  && python3 chtc/validate_condor_submit_lifecycle.py \
       chtc/paper_v_cwrmf_event_guard_accel_20260915/submit_cwrmf_accel_s0*.sub \
  && echo LIFECYCLE_OK'

ssh ap2001 'cd ~/Holstein_paper_v_cwrmf_accel_20260915 \
  && condor_submit -dry-run /tmp/paper_v_accel_s01.ad \
       chtc/paper_v_cwrmf_event_guard_accel_20260915/submit_cwrmf_accel_s01.sub'

ssh ap2001 'cd ~/Holstein_paper_v_cwrmf_accel_20260915 \
  && condor_submit chtc/paper_v_cwrmf_event_guard_accel_20260915/submit_cwrmf_accel_s01.sub'
```

Submit **one segment at a time**: segment `k+1` declares segment `k`'s
`checkpoint.npz` as a transfer input, so it must not be submitted until that
file is back on the AP.

Monitor with `condor_watch_q` on the batch's Condor log file; do not poll
`condor_q` in a loop.

## Collect by receipt, never by name

Each segment stages its own archive into a directory named for the archive's
SHA-256, so two stage-outs can never overwrite one another even when a run id
is reused:

```
archives_by_receipt/<archive_sha256>/paper_v_cwrmf_event_guard_accel_20260915_v1__<run_id>.tar.gz
archives_by_receipt/<archive_sha256>/receipt.json
```

```bash
ssh ap2001 'cd ~/Holstein_paper_v_cwrmf_accel_20260915 \
  && for d in archives_by_receipt/*/; do \
       echo "$d $(python3 -c "import json,sys;r=json.load(open(sys.argv[1]));print(r[\"run_id\"], r.get(\"stop_reason\"), r.get(\"last_time\"), r[\"archive_bytes\"], r[\"driver_exit_status\"])" "$d/receipt.json")"; \
     done'

mkdir -p /tmp/paper_v_accel_arrivals/<sha>
scp -q "ap2001:~/Holstein_paper_v_cwrmf_accel_20260915/archives_by_receipt/<sha>/*" \
  /tmp/paper_v_accel_arrivals/<sha>/
tar -xzf /tmp/paper_v_accel_arrivals/<sha>/*.tar.gz -C /tmp/paper_v_accel_arrivals/<sha>/
```

`receipt.json` records the run id, the four contract arguments, the declared
wall budget, the resolved solver setting profile and seed policy, the archive
SHA-256 and byte count, the driver's exit status, and the two fields that say
what the segment reached: `stop_reason` and `last_time`, copied from the run's
own `summary.json`.

| `stop_reason` | `driver_exit_status` | what to submit next |
|---|---|---|
| `final_time` | 0 | the next segment, from this segment's `checkpoint.npz` |
| `wall_budget` | 0 | **this** segment's `final_time` again, from this segment's `checkpoint.npz`, with a new run id |
| `step_failure` | 1 | nothing until the failure is read: `summary.json` gives the atom's message, and `checkpoint.npz` is still the last accepted step |
| `unrecorded` | nonzero | the driver died before writing `summary.json`; read the job's `.err` |

## Traps

- **The 72-hour wall.** The pool holds any job that has not finished in 72 h,
  and a held job transfers nothing: stdout, stderr and outputs are lost, and
  `condor_transfer_data` does not apply unless the job was submitted with
  `-spool`. Every segment here must exit normally. Never `condor_rm` a held
  job that was submitted with spool. This is what `--maximum-wall-seconds
  138240` is for: it makes the segment end itself at `0.8 x +MaxRuntime`
  rather than run into the wall.
- **A `wall_budget` segment did not reach its `final_time`.** Reading the
  receipt's `driver_exit_status` alone will say a stopped segment succeeded,
  because it did: every step it took was accepted. `stop_reason` is the field
  that separates a finished segment from a stopped one, and a stopped one is
  resubmitted at its **own** `final_time`, not the next segment's.
- **The driver refuses an existing output directory.** It creates
  `--output-directory` with `exist_ok=False`, so a job that restarts after
  eviction into a sandbox that already holds its own partial output would
  crash. That is why `when_to_transfer_output = ON_EXIT` here rather than the
  2026-08-18 lane's `ON_EXIT_OR_EVICT`, and why a resume is a **new submission
  with a new run id**, never a restart of the same job.
- **A resume needs the checkpoint on the AP.** Segment `k+1` lists
  `raw_outputs/<batch>/<segment k run id>/checkpoint.npz` in
  `transfer_input_files`; submit it only after segment `k` has returned.
- **The checkpoint carries the accepted working set only under a carried seed
  policy.** `checkpoint.npz` always carries `bundle_rank_hint` and
  `bundle_accepted_rank`; it carries the 62-row `bundle_accepted_basis` only
  when `CWRMF_BUNDLE_SEED_POLICY` names a policy that reads it. Resuming under
  a different seed policy than the segment that wrote the checkpoint is a
  configuration change, not a continuation.
- **Duo.** `ssh -MNf ap2001` is the user's to run. Every command above reuses
  that master; when it expires everything fails with
  `Permission denied (publickey)`, which is a dead tunnel, not a broken
  package.
