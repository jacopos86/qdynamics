# CWRMF resume completion and the CHTC acceleration lane, 2026-09-15

Worktree `/Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914`,
branch `claude/paper-v-routes-20260914`, from HEAD `d1f4eb83`.  Nothing is
committed and nothing is uploaded or submitted.  Defaults unchanged.

Task: (a) make a checkpoint resume carry the accepted working set, not only
the monotone rank hint; (b) smoke the balanced event-triggered guard with the
round-3 acceleration flags and prove the resume runs with them; (c) build the
dated lane `chtc/paper_v_cwrmf_event_guard_accel_20260915/` with bounded
segments, and package its bundle.

## Negative result first

The round-3 acceleration this lane is named for is a **strict**-profile
result, and the lane runs the **balanced** profile.  Recorded in
`agent_guidance/paper-v/overnight/bundle-rank-seed-20260915.md`, section (c)
part two: over `t = 0` to `0.05` at `h = 0.0025`, balanced, always-on guard,
`--bundle-seed-policy carried_accepted_basis --bundle-seed-rank-margin 2`
costs `879.775 s` of Clarabel solve tier against the balanced default's
`257.439 s` -- `3.42x` slower -- with the maximum readable-rate residual at
`2.8753e-03` against `2.9030e-05` and a peak resident set of `3876 MiB`
against the `4000 MB` guard.  The measured reason is that the balanced
`psd_inflation = 1e-8` lifts the predictor spectrum above the strict entry
threshold, so the balanced tier does not walk its working set; carrying the
working set then only enlarges it.

Consequence for the lane: the submit files default the seed policy to
`monotone_hint` and carry `max_threads_8` alone, which is the only one of the
two accelerations with a balanced-profile qualification.  The carried policy
is available in the lane as a named environment override
(`CWRMF_BUNDLE_SEED_POLICY`, `CWRMF_BUNDLE_SEED_RANK_MARGIN`) and is not the
default of any submit file in it.  See "Lane acceleration decision" below.

## (a) The resume now carries the accepted working set

The gap round 3 recorded: `restore_bundle_rank_hint` restores the monotone
rank hint across a checkpoint resume, but the accepted working set -- the
62-row orthonormal block the adaptive bundle tier accepted on -- was not
saved, so a resumed rollout under `carried_accepted_basis` rebuilt its first
atom's working set from the recorded rule.

### What changed

| file | change |
|---|---|
| `paper_5/src/paper5/stability/apcm_carried_witness.py` | `CWRMF_BASIS_CARRYING_SEED_POLICIES`; `CarriedWitnessModel.bundle_accepted_basis` property; `.restore_bundle_accepted_rank(value)`; `.restore_bundle_accepted_basis(basis, rank=None)` with 62-row and 1--61-column validation |
| `paper_5/src/paper5/stability/apcm_carried_witness_analysis.py` | `checkpoint.npz` gains `bundle_accepted_rank` always and `bundle_accepted_basis` only when the active seed policy reads it; `--initial-state-file` restores both when present |
| `paper_5/tests/test_apcm_carried_witness.py` | 41 -> 45 tests (the three below plus the tiny-resume regression) |

The checkpoint contract is additive: an older `checkpoint.npz` with only
`bundle_rank_hint` still loads, and a default-path (`monotone_hint`)
checkpoint gains one scalar and no array.

### Measured: the resumed rollout reproduces the unbroken one

Strict numerical profile, always-on guard, strong coupling `lambda = 1.5`,
`gamma = 0.5`, drive amplitude 1, phonon cutoff 16, exact-correlated
preparation, `--bundle-seed-policy carried_accepted_basis
--bundle-seed-rank-margin 2`.  "Resumed" means the interval was run as two
driver invocations, the second started with `--initial-state-file` on the
first's `checkpoint.npz`.  "Hint only" is the same resume from a checkpoint
with the `bundle_accepted_basis` key removed, i.e. the behaviour before this
change.  Differences are the maximum absolute difference of the last recorded
row of `autonomous_trajectory.npz` against the unbroken rollout.

`t = 0` to `0.005` at `h = 0.0025` (2 SSPRK2 steps, artifacts under
`output/local_runs/paper_v_cwrmf_resume_20260915_v1/`):

| quantity at `t = 0.005` | resumed, basis carried | resumed, hint only |
|---|---:|---:|
| 31 reported archive coordinates | `0.0` | `0.0` |
| 384 carried coordinates | `0.0` | `2.026859e-08` |
| completion correction norm | `0.0` | `1.361523e-07` |
| readable-rate residual | `0.0` | `3.922484e-08` |
| minimum shifted lower bound | `0.0` | `1.364493e-13` |
| minimum unshifted 62-row Gram eigenvalue | `0.0` | `1.384018e-13` |
| velocity margin | `0.0` | `1.735998e-07` |
| active critical modes at that atom | 35 (unbroken 35) | 36 |
| wall seconds of the resumed segment | `30.4` | `44.4` |

`t = 0` to `0.003` at `h = 0.0015` (2 SSPRK2 steps, the interval the
repository test uses): resumed with the basis carried is bitwise on every
recorded array; hint only differs by `9.317855e-08` on the 384 carried
coordinates, `6.246955e-08` on the completion correction norm and
`1.538335e-07` on the velocity margin, and accepts at 33 active critical
modes against the unbroken 32.

So the resume is now exact under the carried policy, and the difference the
old fallback left is two decades above the round-3 acceptance band it was
compared against, not a roundoff.

### Tests

| test | what it pins |
|---|---|
| `test_accepted_working_set_can_be_restored_for_continuation` | the property is `None`/0 on a fresh model; the two restores set rank and block; a non-62-row block, an empty block and a 62-column block are rejected |
| `test_default_checkpoint_carries_the_scalars_and_no_working_set` | a default-policy (`monotone_hint`) run writes `bundle_rank_hint` and `bundle_accepted_rank` and no `bundle_accepted_basis` |
| `test_resumed_rollout_under_carried_basis_reproduces_the_unbroken_one` | the strict `t = 0` to `0.003` at `h = 0.0015` resumed rollout equals the unbroken one exactly on every recorded array except `maximum_atom_seconds` |

The third test is the discriminating one: it fails on the pre-change
behaviour, by the `9.317855e-08` carried-coordinate difference above.  The
interval is chosen at `h = 0.0015` because at `h = 0.001` the first step is
handled entirely by the critical-Schur tier (6 Clarabel calls, `0.03 s`,
maximum 6 active critical modes) and accepts no bundle solve, so there is
nothing to carry and the test would be vacuous; at `h = 0.0015` the first
step accepts at rank 24.  The three tests run in `56.6 s`.

### One latent resume defect found and fixed

Rehearsing the lane's inner runner exposed a second, unrelated break in the
resume path: `apcm_carried_witness_analysis.main` decided whether a rollout
starts at zero with `np.isclose(trajectory.times[0], 0.0)`, whose default
absolute tolerance is `1e-8`, while `exact_holstein_driven_trajectory` rejects
any first sample time above `1e-15`.  A resume whose start time is at or below
`1e-8` was therefore treated as a rollout from zero and crashed with
`ValueError: sample_times must start at zero` after the rollout had already
completed, losing the scoring but not the checkpoint.  The branch now applies
the same `1e-15` test the reference applies.  Pinned by
`test_resume_below_the_exact_reference_zero_tolerance_still_scores`
(`1.7 s`).  Real segments start at `0.5`, `1.0`, ... and were never affected;
a rehearsal or a sub-`1e-8` restart was.

## (b) Local smoke of the acceleration flag set on the unguarded prefix

Balanced numerical profile, event-triggered guard, `--event-gram-hard-floor
2e-4`, `--maximum-step-subdivisions 2`, `--bundle-seed-policy
carried_accepted_basis`, `--bundle-seed-rank-margin 2`,
`--solver-setting-profile max_threads_8`, `--tier-timing`, strong coupling,
phonon cutoff 16, `t = 0` to `0.05` at `h = 0.0025` (20 SSPRK2 steps).
Artifacts `output/local_runs/paper_v_cwrmf_event_guard_accel_smoke_20260915_v1/`.

| quantity | value |
|---|---:|
| autonomous wall seconds | `2.646` |
| Clarabel solve tier calls / seconds | `0` / `0.0` |
| 62-row Gram eigendecomposition calls / seconds | `61` / `0.0849` |
| unguarded velocity evaluations / seconds | `40` / `0.1174` |
| dynamic all-coordinate scalar RMS against the exact cutoff-16 reference | `3.235917e-06` |
| maximum readable-rate residual | `0.0` |
| maximum active critical modes | 7 |
| driver-process peak resident, MiB | 651 |

The run's own `numerical_settings` block records every requested option:
`guard_policy = event_triggered`, profile `balanced`,
`event_gram_hard_floor = 0.0002`, `maximum_step_subdivisions = 2`,
`bundle_seed_policy = carried_accepted_basis`,
`bundle_seed_rank_margin = 2`, `solver_setting_profile = max_threads_8`.

**Both accelerations are inert over this interval**, and the run says so: the
adaptive bundle tier is entered zero times, so there is no working set to
carry and no Clarabel factorisation to thread.  The smoke proves the flag set
parses, reaches the settings, and runs; it is not evidence that the flags are
fast.  The guard's first trigger in the record is near `t = 0.0825`.

For the horizon: the dynamic all-coordinate scalar RMS `3.235917e-06` over
`t = 0` to `0.05` reproduces the strict always-on anchor's `3.23592e-06` over
the same interval, at `2.646 s` against that anchor's `1,471 s`.

### The resume runs with the same flag set

Resuming from that run's `checkpoint.npz` for one more step: `initial_time`
`0.05` read from the checkpoint, `last_time` `0.0525`, one completed step,
`success`, `0.497 s`, dynamic all-coordinate scalar RMS over the resumed
interval `1.957433e-07`.  The checkpoint carries `bundle_rank_hint = 0` and
`bundle_accepted_rank = 0` and **no** `bundle_accepted_basis`, because no
bundle solve was accepted -- the correct behaviour, and the reason the lane's
resume contract is stated in terms of what the policy read, not what the flag
asked for.

## (c) The lane `chtc/paper_v_cwrmf_event_guard_accel_20260915/`

Mirrors `chtc/paper_v_cwrmf_event_guard_20260818/` and differs in four
declared ways: bounded segments that exit normally and resume from
`checkpoint.npz`, 8 CPUs for the named `max_threads_8` solver setting set,
12 GB for the recorded `6.05 GB` peak, and receipt-named stage-out.  The
2026-08-18 lane is left exactly as written.

| file | role |
|---|---|
| `README.md` | job shape, segment arithmetic, argument contract, upload/extract/validate/submit/collect commands, the five traps |
| `run_cwrmf_event_guard_accel_apptainer.sh` | outer wrapper; `--cleanenv` plus the three named option variables and `OMP_NUM_THREADS` |
| `run_cwrmf_event_guard_accel.sh` | inner runner; flag assembly, conic-stack bootstrap, driver call, receipt-named archive |
| `submit_cwrmf_accel_s01.sub` ... `s08.sub` | one bounded segment each, `t = 0` to `4` in eight steps of `0.5` |

### Argument contract

```
run_id  final_time  time_step  event_gram_hard_floor  initial_state_file
```

`initial_state_file` is the previous segment's `checkpoint.npz` relative to
the job working directory, or the literal `none` for the first segment.  The
driver reads the resume time from the checkpoint, so `final_time` is absolute
in every segment.

### Why the segment is `0.5` time units

The local completion of this exact guard took `14,722 s` for `0.38` time units
of guarded propagation (`t = 0.1175` to `0.5`; the prefix below the trigger is
nearly free), which is `10.8 h` per unit of propagated time on the replacement
M4 Pro.

| segment length | at the local rate | at a 2x execute-node derate | at 4x |
|---:|---:|---:|---:|
| `0.5` | `5.4 h` | `10.8 h` | `21.5 h` |

`+MaxRuntime` is `48 h` and the pool holds at `72 h` with no stage-out, so
`0.5` keeps the whole derate range inside the bound.  The rate beyond
`t = 0.5` is unmeasured -- no guarded CWRMF run has ever passed it -- so the
README instructs splitting the remaining segments if segment 02 returns above
about `12 h`.

### Job shape

Single vanilla job per segment, `request_cpus = 8`, `request_memory = 12GB`,
`request_disk = 30GB`, `+MaxRuntime = 172800`, `requirements = TARGET.HasSIF`,
`when_to_transfer_output = ON_EXIT`, `preserve_relative_paths = True`,
`transfer_output_files = raw_outputs, logs, archives_by_receipt`.

`ON_EXIT`, not the 2026-08-18 lane's `ON_EXIT_OR_EVICT`, and the reason is in
the code: the driver creates `--output-directory` with `exist_ok=False`, so a
job restarted into a sandbox that already carries its own partial output would
crash before propagating anything.  A resume in this lane is a new submission
with a new run id, never a restart of the same job.

### Validators and rehearsal

`chtc/validate_condor_submit_lifecycle.py` is the lane lifecycle validator in
this repository (used by the Paper II protocol at
`agent_guidance/time-dynamics/paper-ii-run-protocol.md` section 3).  All eight
submit files pass it, exit status 0.  No package-level `validate_package.py`
exists for a hand-built lane of this shape; the Paper II one is generated by
its own bundle tool and is specific to that campaign's action list.

The inner runner was rehearsed end to end outside Apptainer, in a scratch
directory with `paper_5` linked in, at `h = 1e-8`:

| rehearsal | result |
|---|---|
| segment 1, `initial_state_file = none` | driver ran with the assembled flags, `success`, archive `12,967` bytes staged at its own SHA-256 |
| segment 2, resuming from segment 1's `checkpoint.npz` | `last_time 2e-08`, `success`, archive `12,818` bytes, receipt records `initial_state_file` |
| segment with a missing checkpoint | refused before the driver ran, exit status 2 |
| both `receipt.json` files | valid JSON carrying campaign, run id, the four contract arguments, resolved solver setting profile and seed policy, archive SHA-256, byte count, driver exit status |

### Bundle

`output/local_runs/paper_v_cwrmf_event_guard_accel_20260915/paper_v_cwrmf_event_guard_accel_20260915.tgz`

| quantity | value |
|---|---|
| SHA-256 | `3110777f8dd1debe73b771c243d2bc3d60eafda4d3576c70b6ba67186261e947` |
| bytes | `500713` |
| entries | 101 |
| contents | `paper_5/src`, `paper_5/pyproject.toml`, the lane directory, `chtc/validate_condor_submit_lifecycle.py` |

The Apptainer image is not in the bundle: `chtc/time_dynamics_optuna/image.sif`
is `216,371,200` bytes and is not a repository file.  It must already be in
the remote submit root, hard-linked from a Paper II root if one holds it.

## Commands for the user, after `ssh -MNf ap2001`

Run them in this order, one segment at a time.  Nothing below has been run by
me: this agent never uploads or submits.

```bash
# 1. make the remote root
ssh ap2001 "mkdir -p ~/Holstein_paper_v_cwrmf_accel_20260915/chtc/time_dynamics_optuna"

# 2. upload the bundle and the image (skip the image if a Paper II root on the
#    AP already holds the same image.sif; hard-link it into this root instead)
scp /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_cwrmf_event_guard_accel_20260915/paper_v_cwrmf_event_guard_accel_20260915.tgz \
  ap2001:~/Holstein_paper_v_cwrmf_accel_20260915/
scp /Users/lordskyler/Holstein_test/chtc/time_dynamics_optuna/image.sif \
  ap2001:~/Holstein_paper_v_cwrmf_accel_20260915/chtc/time_dynamics_optuna/image.sif

# 3. extract, and check the bundle digest against
#    edffe887de74267dd557973df4532383524e559d9dba13d098bfea5844d5110f
#    (the rebuilt bundle; 3110777f... below is the pre-wall-budget one and
#    must not be uploaded -- see the later section of this report)
ssh ap2001 'cd ~/Holstein_paper_v_cwrmf_accel_20260915 \
  && sha256sum paper_v_cwrmf_event_guard_accel_20260915.tgz \
  && tar -xzf paper_v_cwrmf_event_guard_accel_20260915.tgz \
  && chmod +x chtc/paper_v_cwrmf_event_guard_accel_20260915/*.sh \
  && mkdir -p logs raw_outputs archives_by_receipt'

# 4. validate the lifecycle of all eight submit files
ssh ap2001 'cd ~/Holstein_paper_v_cwrmf_accel_20260915 \
  && python3 chtc/validate_condor_submit_lifecycle.py \
       chtc/paper_v_cwrmf_event_guard_accel_20260915/submit_cwrmf_accel_s0*.sub \
  && echo LIFECYCLE_OK'

# 5. dry-run and submit segment 01 only
ssh ap2001 'cd ~/Holstein_paper_v_cwrmf_accel_20260915 \
  && condor_submit -dry-run /tmp/paper_v_accel_s01.ad \
       chtc/paper_v_cwrmf_event_guard_accel_20260915/submit_cwrmf_accel_s01.sub'
ssh ap2001 'cd ~/Holstein_paper_v_cwrmf_accel_20260915 \
  && condor_submit chtc/paper_v_cwrmf_event_guard_accel_20260915/submit_cwrmf_accel_s01.sub'

# 6. when segment 01 has returned, list what arrived and submit segment 02
ssh ap2001 'cd ~/Holstein_paper_v_cwrmf_accel_20260915 && ls -1 archives_by_receipt/'
ssh ap2001 'cd ~/Holstein_paper_v_cwrmf_accel_20260915 \
  && condor_submit chtc/paper_v_cwrmf_event_guard_accel_20260915/submit_cwrmf_accel_s02.sub'

# 7. collect one arrival by receipt
mkdir -p /tmp/paper_v_accel_arrivals/<sha>
scp -q "ap2001:~/Holstein_paper_v_cwrmf_accel_20260915/archives_by_receipt/<sha>/*" \
  /tmp/paper_v_accel_arrivals/<sha>/
tar -xzf /tmp/paper_v_accel_arrivals/<sha>/*.tar.gz -C /tmp/paper_v_accel_arrivals/<sha>/
```

Monitor with `condor_watch_q` on the batch's Condor log file; do not poll
`condor_q` in a loop.  Segment `k+1` declares segment `k`'s `checkpoint.npz`
as a transfer input, so it must not be submitted before that file is on the
AP.

## Recommendation, one decision

The one thing this lane cannot do is bound itself in wall time: the segment
length is the only bound, and the cost per unit time beyond `t = 0.5` has
never been measured.  The cheap fix is a named driver option that stops the
rollout at the last accepted step after a declared number of seconds and exits
normally, turning the `72 h` pool hold from a data-loss event into a shorter
segment.  It is a default-off flag with a default-path test, on the file this
key owns.

-> Recommended: add `--maximum-wall-seconds` to
`apcm_carried_witness_analysis` and wire it into the lane's inner runner as
`0.8 * MaxRuntime` before segment 02 is submitted.  Say yes and it is the next
increment; say no and the lane ships as built, with segment length as the only
bound.

## Verification ledger

| gate | result |
|---|---|
| full `paper_5` suite under `ram_guard.py --limit-mb 4000` | `442 passed, 9 warnings in 265.81 s`, peak `1820 MB` |
| `test_apcm_carried_witness.py` | 45 tests collected (41 before this key) |
| the three resume tests together | `56.6 s`; the tiny-resume regression `1.7 s` |
| `chtc/validate_condor_submit_lifecycle.py` on all eight submit files | exit status 0 |
| inner-runner rehearsal, first segment / resumed segment / missing checkpoint | `success` / `success` / refused with exit 2 |
| bundle | `500713` bytes, SHA-256 `3110777f8dd1debe73b771c243d2bc3d60eafda4d3576c70b6ba67186261e947`, 101 entries |
| impact, read-only from the main checkout's index (18 commits behind) | `restore_bundle_rank_hint` LOW, exact, 4 impacted, sole upstream process `apcm_carried_witness_analysis.main`; the new symbols are not in the index and a text search over `paper_5`, `pipelines`, `src` finds them only in the two owned source files and their test file |

The worktree is shared with the concurrent projective-route key; that key's
changes to `apcm_projective_*`, `projected_apcm.py` and
`cheap_projection_route*.py` are in the same suite run and are not this key's.
Files this key touched: `paper_5/src/paper5/stability/apcm_carried_witness.py`,
`paper_5/src/paper5/stability/apcm_carried_witness_analysis.py`,
`paper_5/tests/test_apcm_carried_witness.py`,
`paper_5/notes/electron_phonon_closure_worklog_20260803.md`, section 3.5 of
`agent_guidance/paper-v/paper-v-state-and-plan-20260914.md`, this report, the
new lane directory, and git-ignored artifacts under
`output/local_runs/paper_v_*`.  Nothing is committed.

# The segment's wall-time bound, 2026-09-15 (later the same day)

Same worktree and branch, from HEAD `5f976077`, the commit carrying everything
above.  The decision the section above asked for was taken: yes.  Nothing is
committed, nothing is uploaded or submitted, and no default of the model or of
the driver changed.

**The bundle digest in "Commands for the user" above is superseded.**  The
bundle was rebuilt with the wall budget in it: upload
`edffe887de74267dd557973df4532383524e559d9dba13d098bfea5844d5110f`, `520921`
bytes, and not `3110777f...`, `500713` bytes.  Every other command in that
list is unchanged.

## What changed

| file | change |
|---|---|
| `paper_5/src/paper5/stability/apcm_carried_witness.py` | `integrate_cwrmf_ssprk2` gains the keyword-only `stop_requested`, default `None`; `_truncated` gains a keyword-only `success`, default `False` |
| `paper_5/src/paper5/stability/apcm_carried_witness_analysis.py` | `--maximum-wall-seconds`, default `None`; the deadline predicate; `maximum_wall_seconds` and `stop_reason` in `summary.json` and in the compact line, written only when the option is set |
| `paper_5/tests/test_apcm_carried_witness.py` | 45 -> 48 tests |
| `chtc/paper_v_cwrmf_event_guard_accel_20260915/run_cwrmf_event_guard_accel.sh` | passes `--maximum-wall-seconds ${CWRMF_MAXIMUM_WALL_SECONDS:-138240}`; copies `stop_reason` and `last_time` out of the run's own `summary.json` into `receipt.json` |
| `chtc/paper_v_cwrmf_event_guard_accel_20260915/run_cwrmf_event_guard_accel_apptainer.sh` | passes `CWRMF_MAXIMUM_WALL_SECONDS` through `--cleanenv` |
| `chtc/paper_v_cwrmf_event_guard_accel_20260915/README.md` | the second bound, the named-option row, the receipt table, two traps |

## The contract

`--maximum-wall-seconds` declares a wall-clock budget for the rollout.  It is
off unless given, and when it is not given the driver consults nothing and
writes exactly the `summary.json` it wrote before.

When it is given, the deadline is the driver's own start plus the budget --
the same clock `autonomous_wall_seconds` reports, so the budget covers the
preparation as well as the rollout.  After every accepted SSPRK2 step except
the last, and after `checkpoint.npz` has already recorded that step, the
driver asks whether the deadline has passed.  At the first accepted step that
ends past it the rollout stops there: `autonomous_trajectory.npz`,
`trajectory.npz` and `summary.json` are written for the steps taken,
`summary.json` records `stop_reason = wall_budget` beside
`maximum_wall_seconds`, `success` is `true` because every step it reports was
accepted, and the process exits 0.  A run that reaches `--final-time` first
records `stop_reason = final_time`; one ended by a failed radial atom or an
unresolved convex node records `step_failure` and exits 1.

The last step is deliberately exempt: it is the step that ends the rollout at
`--final-time` on the ordinary path, and stopping before it would turn a
finished segment into a stopped one.  The offline scoring against the exact
cutoff-16 reference, the artifact writing and the archiving all happen after
the rollout ends, which is why the lane declares `0.8` of `+MaxRuntime` and
not all of it.

At least one step is always taken before the budget is consulted, so a
resubmission always advances the trajectory; a budget shorter than a single
accepted step degenerates to one step per submission, not to no progress.

## Measured

Rehearsal of the lane's inner runner outside Apptainer, in a scratch directory
with `paper_5` linked in, balanced profile, event-triggered guard,
`--event-gram-hard-floor 2e-4`, `--maximum-step-subdivisions 2`,
`--solver-setting-profile max_threads_8`, `h = 1e-8`, the lane's own flag
assembly:

| rehearsal | budget reaching the driver | result |
|---|---|---|
| segment to `t = 2e-8`, no previous checkpoint | `138240 s`, the lane default | `success`, `last_time 2e-08`, `stop_reason final_time`, archive `14466` bytes, exit 0 |
| segment to `t = 3e-8` under `CWRMF_MAXIMUM_WALL_SECONDS=1e-9` | `1e-9 s` | stopped after 1 accepted step at `t = 9.999999999999999e-09`, `stop_reason wall_budget`, archive `13103` bytes, exit 0 |
| that same segment's `t = 3e-8` again, resumed from the checkpoint it left | `138240 s` | `success`, `last_time 3e-08`, `stop_reason final_time`, archive `14706` bytes, exit 0 |
| segment with a missing declared checkpoint | not reached | refused before the driver ran, exit 2 |
| segment whose driver dies before `summary.json` (interval not an integer multiple of the step) | `138240 s` | receipt still written, `stop_reason unrecorded`, `last_time unrecorded`, `driver_exit_status 1` |

The stopped segment's `receipt.json`, abridged:

```json
{
  "run_id": "rehearsal_s02_budget",
  "final_time": "3e-8",
  "maximum_wall_seconds": "1e-9",
  "stop_reason": "wall_budget",
  "last_time": "9.999999999999999e-09",
  "driver_exit_status": 0
}
```

## Tests

| test | what it pins |
|---|---|
| `test_declared_stop_request_halts_the_rollout_at_an_accepted_step` | the predicate is consulted after accepted steps 1 and 2 of a 3-step rollout and never on the last; a rollout it stops reports `success`, one completed step, two recorded times, the caller-stop message, and a final state bitwise equal to the uninterrupted rollout's state at that same step |
| `test_wall_budget_is_off_by_default_and_a_budget_not_reached_is_inert` | the parser default is `None`; a non-positive budget is refused before the output directory is created; over `t = 0` to `2e-8` at `h = 1e-8` a `600 s` budget leaves every recorded array of `autonomous_trajectory.npz` bitwise equal to the unbudgeted run and records `stop_reason final_time`, while the unbudgeted run's summary carries neither new key |
| `test_a_reached_wall_budget_stops_cleanly_and_leaves_a_resume_point` | a `1e-9 s` budget over `t = 0` to `3e-8` at `h = 1e-8` exits 0 with `stop_reason wall_budget`, one completed step, `last_time 1e-8` against `final_time_requested 3e-8`, two recorded times, a `checkpoint.npz` at step 1 carrying the 384 carried coordinates, and a second invocation that resumes from it to `t = 2e-8` |

The three run in `2.3 s`.

## What changes for the user

Only the digest and what to read off a returned segment.

```bash
# step 3 above, with the rebuilt bundle's digest
ssh ap2001 'cd ~/Holstein_paper_v_cwrmf_accel_20260915 \
  && sha256sum paper_v_cwrmf_event_guard_accel_20260915.tgz \
  && tar -xzf paper_v_cwrmf_event_guard_accel_20260915.tgz \
  && chmod +x chtc/paper_v_cwrmf_event_guard_accel_20260915/*.sh \
  && mkdir -p logs raw_outputs archives_by_receipt'
# expect edffe887de74267dd557973df4532383524e559d9dba13d098bfea5844d5110f
```

| `stop_reason` | `driver_exit_status` | submit next |
|---|---|---|
| `final_time` | 0 | the next segment, from this segment's `checkpoint.npz` |
| `wall_budget` | 0 | **this** segment's `final_time` again, from this segment's `checkpoint.npz`, with a new run id |
| `step_failure` | 1 | nothing until the atom's message in `summary.json` is read |
| `unrecorded` | nonzero | nothing; the driver died before `summary.json`, read the job's `.err` |

`driver_exit_status` alone no longer separates a finished segment from a
stopped one: both are 0, and that is the point of the change.

## Verification ledger

| gate | result |
|---|---|
| `paper_5/tests/test_apcm_carried_witness.py` under `ram_guard.py --limit-mb 4000` | `48 passed in 68.85 s`, peak `1753 MB` |
| full `paper_5` suite under the same guard | `466 passed, 9 warnings in 263.21 s`, peak `1680 MB`; the count includes the concurrent projective and cheap-projection keys' tests |
| `chtc/validate_condor_submit_lifecycle.py` on all eight submit files, in the worktree and again on the copies inside the rebuilt bundle | exit status 0 both times |
| inner-runner rehearsal, five cases | the table above |
| bundle | `520921` bytes, SHA-256 `edffe887de74267dd557973df4532383524e559d9dba13d098bfea5844d5110f`, 101 entries, the same four contents as before |
| impact, read-only from the main checkout's index | `integrate_cwrmf_ssprk2` LOW, exact, 4 impacted, sole upstream process `apcm_carried_witness_analysis.main`; a text search over the worktree finds its other caller in `cheap_projection_route.py`, which passes no `stop_requested` and is unaffected by a keyword-only parameter defaulting to `None`; `apcm_carried_witness_campaign.py` and `apcm_carried_witness_stitch.py` read only `numerical_settings` and `parameters` out of a chunk `summary.json` and never pass the option |

Files touched in this increment:
`paper_5/src/paper5/stability/apcm_carried_witness.py`,
`paper_5/src/paper5/stability/apcm_carried_witness_analysis.py`,
`paper_5/tests/test_apcm_carried_witness.py`, the lane's inner runner,
Apptainer wrapper and README,
`paper_5/notes/electron_phonon_closure_worklog_20260803.md`, this report, and
the git-ignored bundle under
`output/local_runs/paper_v_cwrmf_event_guard_accel_20260915/`.  Nothing is
committed.

## Summary

- `--maximum-wall-seconds` exists, is off by default, and is the lane's second
  bound at `138240 s`, which is `0.8 x +MaxRuntime = 172800 s`.
- A stopped segment exits 0 with `stop_reason = wall_budget` and a resume
  point; a finished one exits 0 with `stop_reason = final_time`.
- Bundle `edffe887de74267dd557973df4532383524e559d9dba13d098bfea5844d5110f`,
  `520921` bytes, supersedes `3110777f...`, `500713` bytes.
- 48 carried-witness tests, 466 in the `paper_5` suite, eight submit files
  lifecycle-clean, five inner-runner rehearsals.
- For the user, only the digest changed; the receipt now says whether the next
  submission is the next segment or the same one again.

# The lane's resume is executable, and the bundle names a commit, 2026-09-15 (task 3)

Same worktree and branch, from HEAD `e908c78a`.  Repairs the two defects the
reviewer recorded against this lane in
`agent_guidance/paper-v/overnight/review-routes-20260915b.md` section 3
(corrections 14, 15, 16 and 17).  Nothing is committed, nothing is uploaded or
submitted, and no default of the model or of the driver changed.  No file
under `paper_5/` was touched in this increment.

## Negative result first

**The bundle digest changes again, and the two earlier ones must not be
uploaded.**  Upload
`eed32f13d55ca8a5c036da3e238d826cdaebde9a0e1fd65f57edda391b7a59ff`,
`502619` bytes, 105 entries.  `edffe887...`/`520921` and `3110777f...`/`500713`
are both superseded: the first pinned a working tree carrying three other
keys' uncommitted edits, and neither corresponds to a commit.

**And the reviewer's reading of the shipped submit files was right.**  A
`wall_budget` return could not be continued without hand edits, and the README
said the opposite.  Four fields had to change, not none: the run id in
`arguments`, the fifth argument `initial_state_file`, `transfer_input_files`,
and the `output`/`error`/`+JobBatchName` names that embed the run id.

## (1) The continuation is generated, not hand-edited

`chtc/paper_v_cwrmf_event_guard_accel_20260915/make_resume_submit.py` reads a
returned segment's `receipt.json` (or the run's own `summary.json`, from which
it takes the run id and campaign off the path) and writes the one submit file
that continues from it.

| `stop_reason` read | what it writes | run id | `final_time` | checkpoint named |
|---|---|---|---|---|
| `wall_budget` | a resume of the same segment | `<segment>_r01`, then `_r02`, ... | the stopped segment's own | that segment's own |
| `final_time` | the next segment | `accel_s<k+1>_t<start>_t<end>` | the previous plus `0.5` | that segment's own |
| `step_failure`, `unrecorded` | nothing, exit 2 with the reason | --- | --- | --- |

Naming the returned segment's **own** checkpoint in both cases is what repairs
the second half of the defect: once any segment has been resumed, the run id
holding the checkpoint is no longer the one the shipped `s0(k+1).sub` names.

### A run id cannot be reused

Two refusals, at the two places a reuse can be introduced.

| where | what it refuses | status |
|---|---|---|
| `make_resume_submit.py` | a run id that any `raw_outputs/<campaign>/` directory or any `arguments` line of a `.sub` in the lane already names | exit 2, nothing written |
| `run_cwrmf_event_guard_accel.sh` | a run id that already owns `raw_outputs/<campaign>/<run_id>` in the sandbox, checked before the conic bootstrap and before the driver | exit 3 |

The runner's check is the one that matters at the execute node: a resume
submitted into the run id that wrote the checkpoint receives that checkpoint
into its own output directory under `preserve_relative_paths`, and the
stage-out would then replace the access point's copy with a shorter one.
`archives_by_receipt/<sha>/` is content addressed and was never at risk;
`raw_outputs/<campaign>/<run_id>/`, which holds `checkpoint.npz`, is named and
was.  The README now says so in those terms rather than claiming two
stage-outs can never overwrite one another "even when a run id is reused".

### A stale checkpoint path is refused rather than silently obeyed

The argument contract gains a sixth field, `expected_initial_time`, which the
runner passes to the driver as `--initial-time`.  The driver already compares
that against the checkpoint's own recorded `time` and raises `initial time
disagrees with checkpoint`; no driver change was needed.  The comparison is
`np.isclose` at its default `rtol = 1e-5`, so it separates a segment-scale
mismatch and not roundoff.  The shipped `s02` ... `s08` now declare `0.50`,
`1.00`, ..., `3.50`; `s01` declares `none`.  Declaring a nonzero start with
`initial_state_file = none` is refused by the runner, because that would start
the prepared ground state at that time, which is a different physical run.

### Rehearsed end to end at `h = 1e-8`

Outside Apptainer, in a scratch root with `paper_5` linked in and the lane
copied in; one bounded segment is two steps of `1e-8`, the campaign final time
is `6e-8`, and run-id time labels are at scale `1e11`, so `t = 2e-8` reads
`t2000` exactly as `t = 0.5` reads `t0500` at the lane's own scale of `1000`.
The harness and its log are
`output/local_runs/paper_v_cwrmf_accel_resume_rehearsal_20260915/rehearse_chain.sh`
and `rehearsal.log`.

| step | what was run | result |
|---|---|---|
| segment 01 | `run_id rehearsal_s01_t0000_t2000`, `final_time 2e-8`, no checkpoint, `CWRMF_MAXIMUM_WALL_SECONDS=1e-9` | exit 0, `stop_reason wall_budget`, `last_time 1e-08`, archive `13097` bytes |
| generate | that receipt, defaults | `kind=resume`, `run_id rehearsal_s01_t0000_t2000_r01`, `final_time 2e-08`, checkpoint `.../rehearsal_s01_t0000_t2000/checkpoint.npz`, `expected_initial_time 1e-08` |
| the resume | the generated file's own `arguments`, lane default budget | exit 0, `stop_reason final_time`, `initial_time 1e-08`, `last_time 2e-08` |
| generate | the resume's receipt | `kind=next`, `run_id rehearsal_s02_t2000_t4000`, `final_time 4e-08`, checkpoint `.../rehearsal_s01_t0000_t2000_r01/checkpoint.npz`, `expected_initial_time 2e-08` |
| segment 02 | the generated file's own `arguments` | exit 0, `stop_reason final_time`, `initial_time 2e-08`, `last_time 4e-08` |

The three `autonomous_trajectory.npz` files tile the interval with no gap and
no overlap: `[0, 1e-8]`, `[1e-8, 2e-8]`, `[2e-8, 3e-8, 4e-8]`.  Segment 02's
recorded start is the resume's end, which is the point of the chain: it
started from segment 01's final checkpoint, not from the stopped one.

Five refusals were rehearsed in the same pass:

| refusal | status |
|---|---|
| the runner, on a run id that already owns a `raw_outputs` directory | 3 |
| the generator, asked for a next segment whose run id is already used | 2 |
| the driver, given segment 01's stopped checkpoint under `expected_initial_time 2e-8` | `ValueError: initial time disagrees with checkpoint`, driver exit 1, receipt still written with `stop_reason unrecorded` |
| the runner, given a nonzero `expected_initial_time` with no checkpoint | 2 |
| the generator, given a `step_failure` receipt with no explicit `--kind` | 2 |

`chtc/validate_condor_submit_lifecycle.py` passes on all ten submit files in
the rehearsal lane -- the eight shipped and the two generated -- exit status 0,
and again on the eight in the worktree and the eight inside the rebuilt
bundle.  Narrowing the earlier ledger row, as the reviewer asked: that
validator checks bounded factories and the reserved ordinary-held release
mode.  It does not read `transfer_input_files`, `transfer_output_files`,
`when_to_transfer_output` or the argument contract, so passing it says nothing
about whether a continuation names the right checkpoint; the two run-id
refusals and `expected_initial_time` are what cover that.

## (2) The bundle is built from a commit, and the build is reproducible

Checked with `git status --short` before building, as instructed.  At build
time `paper_5/src`, `paper_5/pyproject.toml` and
`chtc/validate_condor_submit_lifecycle.py` were all clean against HEAD
`e908c78a`: the three keys whose uncommitted edits the reviewer found inside
`edffe887...` have since committed them (`c526c949`, `b04ec23b`, `e908c78a`).
The bundle is nevertheless built the safe way rather than from the working
tree, because this worktree is shared and was in fact edited again during this
increment (`paper_5/tests/test_cheap_projection_route.py`, another key, not a
bundle member).

**Method used: `git archive HEAD` for everything except this lane, plus the
lane directory from the working tree.**  The lane is taken from the working
tree because it is this key's own uncommitted work and is the thing being
shipped.  `chtc/paper_v_cwrmf_event_guard_accel_20260915/build_bundle.py`
prints `git status --short` for both halves on every build, so the report can
name them:

| half | source | state at build |
|---|---|---|
| `paper_5/src` (84 files), `paper_5/pyproject.toml`, `chtc/validate_condor_submit_lifecycle.py` | `git archive e908c78a` | clean against HEAD, verified with `git diff --quiet HEAD` on each of the three |
| `chtc/paper_v_cwrmf_event_guard_accel_20260915/` (13 files) | working tree | 10 modified, 2 untracked (`make_resume_submit.py`, `build_bundle.py`), 1 unchanged |

Archive metadata is normalized -- names sorted, every mtime the HEAD commit's
author time, uid/gid 0 with no owner names, modes `0644` or `0755` by shebang,
gzip written with no timestamp -- so the same commit and the same lane bytes
reproduce the digest.  Rebuilt three times here; identical each time.  The
earlier `tar`-based build was not: two consecutive builds of identical inputs
differed by one byte.

| quantity | value |
|---|---|
| SHA-256 | `eed32f13d55ca8a5c036da3e238d826cdaebde9a0e1fd65f57edda391b7a59ff` |
| bytes | `502619` |
| entries | 105 |
| built from | `git archive e908c78a` plus the lane working tree |

Extracted and diffed against the sources: the lane, `paper_5/src`,
`paper_5/pyproject.toml` and the validator are each byte-identical, and the
lane's submit files pass the lifecycle validator from inside the bundle.

## What changes for the user

The digest, and one new step in the per-segment loop.  Step 3 of "Commands for
the user" now expects:

```bash
ssh ap2001 'cd ~/Holstein_paper_v_cwrmf_accel_20260915 \
  && sha256sum paper_v_cwrmf_event_guard_accel_20260915.tgz \
  && tar -xzf paper_v_cwrmf_event_guard_accel_20260915.tgz \
  && chmod +x chtc/paper_v_cwrmf_event_guard_accel_20260915/*.sh \
  && mkdir -p logs raw_outputs archives_by_receipt'
# expect eed32f13d55ca8a5c036da3e238d826cdaebde9a0e1fd65f57edda391b7a59ff
```

Every segment after the first is now generated from the previous segment's
receipt rather than submitted from a shipped file, and this one command
replaces step 6:

```bash
ssh ap2001 'cd ~/Holstein_paper_v_cwrmf_accel_20260915 \
  && NEXT=$(python3 chtc/paper_v_cwrmf_event_guard_accel_20260915/make_resume_submit.py \
              archives_by_receipt/<sha>/receipt.json | tail -1) \
  && python3 chtc/validate_condor_submit_lifecycle.py "$NEXT" \
  && condor_submit -dry-run /tmp/paper_v_accel_next.ad "$NEXT" \
  && condor_submit "$NEXT"'
```

It prints `kind=<resume|next> run_id=<fresh id> final_time=<t>` and the path it
wrote.  Submitting `submit_cwrmf_accel_s01.sub` by hand is still how the
campaign starts.  The `stop_reason` table's third column is now that command in
both continuing rows, instead of a description of edits to make.

## Verification ledger

| gate | result |
|---|---|
| `paper_5/tests/test_apcm_carried_witness.py` under `ram_guard.py --limit-mb 3000` | `48 passed in 65.52 s`, peak `1671 MB` |
| rehearsal of the full chain, five runs and five refusals, under the same guard | the two tables above; `exit=0`, peak `658 MB` on the first pass and `112 MB` on the repeat (the guard samples at 2 s and each run is sub-second) |
| `chtc/validate_condor_submit_lifecycle.py` | exit 0 on the ten rehearsal submit files, the eight in the worktree, and the eight inside the bundle |
| bundle rebuilt three times from the same inputs | `eed32f13...`, `502619` bytes, 105 entries, identical each time |
| bundle extracted and diffed against its sources | lane, `paper_5/src`, `paper_5/pyproject.toml`, validator all byte-identical |
| `git diff --quiet HEAD` on `paper_5/src`, `paper_5/pyproject.toml`, `chtc/validate_condor_submit_lifecycle.py` | clean, so the committed half of the bundle is HEAD `e908c78a` |
| impact | no `paper_5` symbol was touched; the files changed are the lane's shell, submit and Python files, which no repository symbol imports |

Files touched in this increment, all inside
`chtc/paper_v_cwrmf_event_guard_accel_20260915/`: `make_resume_submit.py`
(new), `build_bundle.py` (new), `run_cwrmf_event_guard_accel.sh`, `README.md`,
`submit_cwrmf_accel_s01.sub` ... `s08.sub`; plus this report and the
git-ignored artifacts under
`output/local_runs/paper_v_cwrmf_accel_resume_rehearsal_20260915/` and
`output/local_runs/paper_v_cwrmf_event_guard_accel_20260915/`.  Nothing is
committed.

## Summary

- A `wall_budget` return is continued by `make_resume_submit.py <receipt>`,
  which writes the resume submit file with a fresh run id, the stopped
  segment's own `final_time`, its own checkpoint as the transfer input, and the
  time that checkpoint holds; the same command writes the next segment from a
  `final_time` receipt, naming that segment's own checkpoint.
- A run id cannot be reused: the generator refuses one already named by a
  `raw_outputs` directory or a lane `.sub`, and the runner refuses at the
  execute node with exit 3 before any work.
- The sixth contract argument `expected_initial_time` makes the driver refuse a
  stale checkpoint instead of silently starting at the wrong time.
- The chain was rehearsed at `h = 1e-8`: stopped segment 01, generated resume,
  resume to segment 01's own `final_time`, generated segment 02 from the
  resume's checkpoint; the trajectories tile `t = 0` to `4e-8` with no gap or
  overlap.  Five refusals rehearsed; ten submit files lifecycle-clean.
- Bundle `eed32f13d55ca8a5c036da3e238d826cdaebde9a0e1fd65f57edda391b7a59ff`,
  `502619` bytes, 105 entries, built from `git archive e908c78a` plus the lane
  working tree, reproducible on rebuild, superseding `edffe887...`/`520921`.
