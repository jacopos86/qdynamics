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
#    3110777f8dd1debe73b771c243d2bc3d60eafda4d3576c70b6ba67186261e947
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
