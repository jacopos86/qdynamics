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

| file | role |
|---|---|
| `run_cwrmf_event_guard_accel_apptainer.sh` | outer wrapper; `--cleanenv` plus the four named option variables and `OMP_NUM_THREADS` |
| `run_cwrmf_event_guard_accel.sh` | inner runner; run-id and checkpoint refusals, flag assembly, conic-stack bootstrap, driver call, receipt-named archive |
| `make_resume_submit.py` | writes the continuation submit file from a returned segment's `receipt.json`, under a run id nothing here has used |
| `build_bundle.py` | builds the upload bundle from `git archive HEAD` plus this lane's working tree, with normalized archive metadata so the digest is reproducible |
| `rehearse_bundle.sh` | the pre-upload gate: extracts a bundle into a scratch root inside no checkout and runs the inner runner there from the bundle's contents alone; see "Pre-upload gate" |
| `submit_cwrmf_accel_s01.sub` ... `s08.sub` | the planned grid, `t = 0` to `4` in eight `0.5`-unit segments; `s01`'s run id `accel_s01_t0000_t0500` was consumed by cluster 10554184, so `s01.sub` is not submitted again |
| `submit_cwrmf_accel_accel_s01_t0000_t0500_r01_20260916.sub` | the retry of segment 01 after cluster 10554184, generated with `--kind retry` from `s01.sub`; run id `accel_s01_t0000_t0500_r01`; the file that starts the campaign |
| `submission_receipt_10554184.json` | the record of the consumed first submission |

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
Its continuation is that same `final_time` again, resumed from the checkpoint
it left, under a fresh run id -- and that is four changes to a submit file,
not none: the run id in `arguments`, the fifth argument `initial_state_file`
(which in every shipped file names the **previous** segment's checkpoint, and
in `s01` the literal `none`), the sixth argument `expected_initial_time`, and
`transfer_input_files`; the `output`, `error` and `+JobBatchName` names carry
the run id too.  `make_resume_submit.py` writes that file instead of asking
for the edits -- see "Continuing the campaign" below.  `stop_reason` and
`last_time` in `receipt.json` are what say which case a returned segment is.

`CWRMF_MAXIMUM_WALL_SECONDS` overrides the budget for one segment.  A shorter
budget is the right response to a segment that returns `wall_budget` with very
little propagated: it makes the next attempt's stop-and-stage cycle shorter,
not slower.  `make_resume_submit.py --maximum-wall-seconds <s>` puts it in the
continuation's `environment` line.

## Argument contract

```
run_id  final_time  time_step  event_gram_hard_floor  initial_state_file \
  expected_initial_time
```

`initial_state_file` is the `checkpoint.npz` the segment resumes from,
relative to the job working directory, or the literal `none` for the first
segment.  For the next segment it is the previous segment's checkpoint; for a
resume of a stopped segment it is that segment's own.  The driver reads the
resume time from the checkpoint, so `final_time` is always absolute.

`expected_initial_time` is the time that checkpoint is supposed to hold, or
`none` to declare nothing.  The runner passes it to the driver as
`--initial-time`, and the driver refuses a checkpoint whose recorded time
disagrees (`ValueError: initial time disagrees with checkpoint`, driver exit
1).  The comparison is `np.isclose` at its default `rtol = 1e-5`, so it
separates a segment-scale mismatch -- resuming from a superseded checkpoint at
`t = 0.31` when `t = 0.50` was meant -- and not roundoff.  Declaring a nonzero
time with `initial_state_file = none` is refused by the runner (exit 2),
because that would start the prepared ground state at that time, which is a
different physical run.

The eight shipped submit files carry the contract for the planned grid; do not
hand-edit a segment's profile, cutoff, coupling, drive, hard floor or step,
and do not hand-edit a continuation -- generate it.

## Continuing the campaign

`make_resume_submit.py` turns a returned segment's `receipt.json` (or the
run's own `summary.json`) into the submit file that continues from it.  It
reads `stop_reason` and writes the one continuation that reason calls for:

| `stop_reason` | what it writes | run id | `final_time` |
|---|---|---|---|
| `wall_budget` | a resume of the same segment, from its own checkpoint | `<segment>_r01`, then `_r02`, ... | the stopped segment's own |
| `final_time` | the next segment | `accel_s<k+1>_t<start>_t<end>` | the previous plus `0.5` |
| `step_failure`, `unrecorded`, `not_returned` | nothing on its own; it refuses and says why | --- | --- |
| any of those three, with `--kind retry` given explicitly | the same segment again, from the input it was given (`none none` for segment 01; the previous segment's checkpoint otherwise) | `<segment>_r<NN>`, the same counter | the segment's own |

The `_r<NN>` counter counts every re-submission of a segment, a resume from
its own checkpoint and a retry from its input alike; the receipt's
`initial_state_file` says which one a run was.  `not_returned` is what a
submit file given as the source stands for: a submission that came back with
nothing, as cluster 10554184 did, has no receipt, so the file it was
submitted from is the source of its retry:

```bash
python3 chtc/paper_v_cwrmf_event_guard_accel_20260915/make_resume_submit.py --kind retry \
    chtc/paper_v_cwrmf_event_guard_accel_20260915/submit_cwrmf_accel_s01.sub
```

A retry of a segment that left a checkpoint is refused: `wall_budget` and
`final_time` receipts are continued, not retried.

For a resume and for the next segment the checkpoint it names is the
**returned segment's own**, which
is what makes a resumed segment's successor correct: `s02`'s shipped file
names `accel_s01_t0000_t0500/checkpoint.npz`, and if segment 01 actually
finished under the run id `accel_s01_t0000_t0500_r01` then that shipped file
is the wrong one to submit.

```bash
# on the AP, from the submit root, after a segment has returned
ssh ap2001 'cd ~/Holstein_paper_v_cwrmf_accel_20260915 \
  && python3 chtc/paper_v_cwrmf_event_guard_accel_20260915/make_resume_submit.py \
       archives_by_receipt/<sha>/receipt.json'
# it prints  kind=<resume|next> run_id=<fresh id> final_time=<t>
# and the path it wrote, which is the file to validate and submit.
```

A run id is used **once**.  The generator refuses to write one that any
`raw_outputs/<campaign>/` directory or any `arguments` line in this lane
already names, and the inner runner refuses at the execute node (exit 3,
before the conic bootstrap and before the driver) when its run id already owns
a directory under `raw_outputs/<campaign>/`.  That second check is what stops
the destructive case: a resume submitted into the run id that wrote the
checkpoint would receive that checkpoint into its own output directory and
then replace it on the access point with a shorter one.

Useful options: `--kind resume|next|retry` overrides what `stop_reason`
chose (`retry` is never chosen automatically);
`--maximum-wall-seconds` shortens the continuation's budget;
`--dry-run` prints the file instead of writing it; `--segment-length`,
`--campaign-final-time` and `--time-label-scale` are the grid arithmetic and
default to `0.5`, `4.0` and `1000`.  A non-default
`CWRMF_SOLVER_SETTING_PROFILE` or `CWRMF_BUNDLE_SEED_POLICY` recorded in the
receipt is restated in the continuation's `environment` line, because resuming
under a different one is a configuration change, not a continuation.

## Job shape

Single vanilla job per segment.  8 CPUs (the named `max_threads_8` Clarabel
setting set, which is why this lane is not the 2026-08-18 lane's 1 CPU),
12 GB memory against the recorded `6.05 GB` peak, 30 GB disk (image, scratch
venv, trajectory and checkpoint), `+MaxRuntime = 172800`, against which the
inner runner passes the driver a `138240 s` wall budget.

The import chain is self-contained -- `paper_5/src` plus numpy, scipy, cvxpy
and clarabel -- only since `paper5/repo.py` resolves the parent repository
lazily.  Before that, `paper5/__init__.py` imported `paper5.repo`, whose
module-level `REPO_ROOT = find_repo_root()` raised
`RuntimeError("Could not locate parent repo root with src/quantum.")` on any
tree without the parent's `src/quantum`, which is every bundle; cluster
10554184 died at import on it after two local rehearsals had passed with
`paper_5` linked into a checkout.  The statement is now enforced rather than
asserted: `rehearse_bundle.sh` proves it for each bundle (see "Pre-upload
gate").  The inner runner bootstraps a scratch venv with pinned versions only
when the time-dynamics image lacks the conic stack.

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
byte count are in `bundle.sha256` and `bundle.bytes` beside it, and its member
list in `bundle.entries`.  The Apptainer image is **not** in the bundle
(206 MB, not a repository file).

`build_bundle.py` in this directory rebuilds it.  Everything the bundle
carries that is not this lane -- `paper_5/src`, `paper_5/pyproject.toml`,
`chtc/validate_condor_submit_lifecycle.py` -- comes from `git archive HEAD`,
never from the working tree, because this worktree is shared with other keys
whose in-flight edits must not reach a remote submit root.  This lane
directory is taken from the working tree, since it is the uncommitted work
being shipped; the build prints `git status --short` for both halves so the
report can name exactly which files that is.  Archive metadata is normalized
(names sorted, every mtime the HEAD commit's author time, uid/gid 0, modes
`0644`/`0755`, gzip with no timestamp), so a rebuild from the same commit and
the same lane bytes reproduces the digest.

### Pre-upload gate: rehearse the bundle, not the worktree

Required before every upload, on the exact bundle that will be uploaded, in
this order: commit the `paper_5/src` the bundle will carry, `build_bundle.py`,
`rehearse_bundle.sh`, upload.

```bash
PATH=/Users/lordskyler/Holstein_test/.venv/bin:$PATH \
  python3 pipelines/shell/ram_guard.py --limit-mb 4000 -- \
  bash chtc/paper_v_cwrmf_event_guard_accel_20260915/rehearse_bundle.sh
# the last line must be  BUNDLE_REHEARSAL_OK <sha256>  with the sha256 in bundle.sha256
```

`rehearse_bundle.sh` extracts the bundle into a scratch root that lies inside
no checkout (`mktemp -d` under `$TMPDIR`; it refuses a root under any
directory holding `src/quantum/pauli_polynomial_class.py`, and a root inside
this worktree), checks that nothing in the extracted tree is a symlink, and
runs from that root with `PYTHONPATH=paper_5/src` only: the import gate
(`import paper5` and the driver, with `paper5.__file__` resolved inside the
scratch root), the driver's `--help`, segment 01 at `h = 1e-8`, the
continuation `make_resume_submit.py` writes from segment 01's receipt, the
negative case of a driver that dies before its output directory exists
(which must still leave a receipt with `stop_reason unrecorded`), the
lifecycle validator on every submit file in the extracted lane, and the
tiling of the two trajectories.  On success it appends the digest to
`bundle.rehearsed` beside the bundle; the last line of `bundle.rehearsed`
must carry the digest in `bundle.sha256`, or the bundle is not the one that
was rehearsed.  It refuses to start on a `python3` without numpy, scipy,
cvxpy and clarabel, so the runner's pip bootstrap can never run locally.

What it catches that the earlier rehearsals did not: those ran the inner
runner in a scratch directory with `paper_5` **linked** into a checkout, so
`paper5.repo`'s walk over the *resolved* ancestors of `repo.py` found the
parent repository and passed, and cluster 10554184 then died at import on the
execute node, where there is no parent.  Run on the bundle `eed32f13...`
that job was submitted from, this gate fails at the import gate with that
job's exact error.

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

The remote root already holds the `eed32f13...` extraction that cluster
10554184 ran from.  Extracting the rehearsed bundle over it replaces every
shipped file; no shipped file was removed between the two bundles, so nothing
stale remains, and whatever empty `raw_outputs/` came back from that job
names no run id.  The campaign starts on the retry file, not on `s01.sub`.

```bash
ssh ap2001 'cd ~/Holstein_paper_v_cwrmf_accel_20260915 \
  && sha256sum paper_v_cwrmf_event_guard_accel_20260915.tgz \
  && tar -xzf paper_v_cwrmf_event_guard_accel_20260915.tgz \
  && chmod +x chtc/paper_v_cwrmf_event_guard_accel_20260915/*.sh \
  && mkdir -p logs raw_outputs archives_by_receipt'
# the sha256sum line must equal bundle.sha256 and the last line of bundle.rehearsed

ssh ap2001 'cd ~/Holstein_paper_v_cwrmf_accel_20260915 \
  && python3 chtc/validate_condor_submit_lifecycle.py \
       chtc/paper_v_cwrmf_event_guard_accel_20260915/*.sub \
  && echo LIFECYCLE_OK'

ssh ap2001 'cd ~/Holstein_paper_v_cwrmf_accel_20260915 \
  && condor_submit -dry-run /tmp/paper_v_accel_s01_r01.ad \
       chtc/paper_v_cwrmf_event_guard_accel_20260915/submit_cwrmf_accel_accel_s01_t0000_t0500_r01_20260916.sub'

ssh ap2001 'cd ~/Holstein_paper_v_cwrmf_accel_20260915 \
  && condor_submit chtc/paper_v_cwrmf_event_guard_accel_20260915/submit_cwrmf_accel_accel_s01_t0000_t0500_r01_20260916.sub'
```

Submit **one segment at a time**: segment `k+1` declares segment `k`'s
`checkpoint.npz` as a transfer input, so it must not be submitted until that
file is back on the AP.

Every segment after the first is submitted the same way, on the file
`make_resume_submit.py` wrote from the previous segment's receipt: generate,
validate, dry-run, submit.

```bash
ssh ap2001 'cd ~/Holstein_paper_v_cwrmf_accel_20260915 \
  && NEXT=$(python3 chtc/paper_v_cwrmf_event_guard_accel_20260915/make_resume_submit.py \
              archives_by_receipt/<sha>/receipt.json | tail -1) \
  && python3 chtc/validate_condor_submit_lifecycle.py "$NEXT" \
  && condor_submit -dry-run /tmp/paper_v_accel_next.ad "$NEXT" \
  && condor_submit "$NEXT"'
```

`validate_condor_submit_lifecycle.py` checks bounded factories and the
reserved ordinary-held release mode.  It does **not** read
`transfer_input_files`, `transfer_output_files`, `when_to_transfer_output` or
the argument contract, so passing it is not a statement that the continuation
names the right checkpoint; `expected_initial_time` and the run-id refusals
are what cover that.

Monitor with `condor_watch_q` on the batch's Condor log file; do not poll
`condor_q` in a loop.

## Collect by receipt, never by name

Each segment stages its own archive into a directory named for the archive's
SHA-256, so two stage-outs can never overwrite one another:

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

That content addressing holds for `archives_by_receipt/` and **not** for
`raw_outputs/<campaign>/<run_id>/`, which is where `checkpoint.npz` lives and
which is named, not hashed: any stage-out reusing a run id would replace it.
That is why a run id is used once and why both the generator and the inner
runner refuse a reused one.  The returned `raw_outputs` also carries the
segment's input checkpoint back unchanged, under the run id that wrote it;
the bytes are identical, so that overwrite is a no-op.

`receipt.json` records the run id, the six contract arguments, the declared
wall budget, the resolved solver setting profile, seed policy and rank margin,
the archive SHA-256 and byte count, the driver's exit status, and the three
fields copied from the run's own `summary.json` that say what the segment did:
`initial_time`, `last_time` and `stop_reason`.

| `stop_reason` | `driver_exit_status` | what to submit next |
|---|---|---|
| `final_time` | 0 | the next segment: `make_resume_submit.py <this receipt>` writes it, from this segment's `checkpoint.npz` |
| `wall_budget` | 0 | **this** segment's `final_time` again: the same command writes it, from this segment's `checkpoint.npz`, under a fresh run id |
| `step_failure` | 1 | nothing until the failure is read: `summary.json` gives the atom's message, and `checkpoint.npz` is still the last accepted step. The generator refuses this receipt until `--kind` is given explicitly |
| `unrecorded` | nonzero | the driver died before writing `summary.json`; read the job's `.err`. The receipt is written even when the driver died before creating its output directory: the runner archives an empty run directory instead of dying at `tar`, which is what cluster 10554184's runner did (exit 2, nothing staged). The generator refuses this receipt unless `--kind retry` is given, and a retry restarts from the input this segment was given |

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
- **A shipped `s0k.sub` is the planned grid, not the continuation.** Once any
  segment has been resumed, the run id that holds the checkpoint is no longer
  the one `s0(k+1).sub` names, and submitting that file would start the
  segment from the stopped, earlier checkpoint. The sixth argument catches it
  -- the driver refuses the pair and exits 1 -- but the file to submit is the
  one `make_resume_submit.py` writes from the receipt, and it is the only
  continuation this lane is meant to run.
- **A run id is consumed even by a failure.** A segment that died still left
  `raw_outputs/<campaign>/<run_id>/` on the AP, or at least its run id in a
  shipped `.sub`, so the retry cannot reuse that run id: the generator refuses
  it, and the runner would refuse it at exit 3. Generate the retry with
  `--kind retry`, which names a fresh one: cluster 10554184 consumed
  `accel_s01_t0000_t0500`, and its retry is `accel_s01_t0000_t0500_r01` in
  `submit_cwrmf_accel_accel_s01_t0000_t0500_r01_20260916.sub`.
- **A rehearsal inside a checkout is not a rehearsal of the bundle.**
  `paper5.repo` climbs the resolved ancestors of `paper_5/src/paper5/repo.py`
  looking for `src/quantum/pauli_polynomial_class.py`; a copy under a
  checkout's `output/`, or a symlink to the worktree's `paper_5`, finds the
  parent and passes what the execute node fails. `rehearse_bundle.sh` refuses
  such a root and is the only rehearsal that counts as the pre-upload gate.
- **The driver refuses an existing output directory.** It creates
  `--output-directory` with `exist_ok=False`, so a job that restarts after
  eviction into a sandbox that already holds its own partial output would
  crash. That is why `when_to_transfer_output = ON_EXIT` here rather than the
  2026-08-18 lane's `ON_EXIT_OR_EVICT`, and why a resume is a **new submission
  with a new run id**, never a restart of the same job.
- **A resume needs the checkpoint on the AP.** Segment `k+1` lists
  `raw_outputs/<campaign>/<segment k run id>/checkpoint.npz` in
  `transfer_input_files`; submit it only after segment `k` has returned. The
  generator refuses to write a continuation whose checkpoint is not in the
  submit root yet, unless `--allow-missing-checkpoint` is given.
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
