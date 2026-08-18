# Paper V CWRMF event-guard CHTC lane (2026-08-18)

First CHTC lane for Paper V. Runs the 384-coordinate carried-witness
(CWRMF) strong-coupling cutoff-16 pilot with the event-triggered Gram
guard: balanced numerical profile, `event_gram_hard_floor = 2e-4` so the
conic repair always fires at a depth it is proven to handle (the
2026-08-18 local balanced attempt hit `NumericalError` only at
`-5.9e-4`, after succeeding at `-2.3e-4`).

The import chain is self-contained: `paper_5/src` plus numpy, scipy,
cvxpy, and clarabel. The inner runner bootstraps a scratch venv with
pinned versions only when the time-dynamics image lacks the conic stack.

Job shape: single vanilla job, 1 CPU (Clarabel is single-thread by
declared contract), 8GB memory (local peaks were 0.6--3.3 GB), 20GB
disk (image + venv + outputs), 12 h MaxRuntime for the expected 2--3 h
`t=0.5` run. The driver checkpoints after every accepted SSP step into
the transferred output directory, so an evicted or held job resumes with
`--initial-state-file` on `checkpoint.npz`.

## Upload from local macOS (repo root)

```bash
COPYFILE_DISABLE=1 tar --no-xattrs \
  --exclude='__pycache__' --exclude='*.egg-info' \
  -czf /tmp/paper_v_cwrmf_event_guard_20260818.tgz \
  paper_5/src paper_5/pyproject.toml chtc/paper_v_cwrmf_event_guard_20260818
scp /tmp/paper_v_cwrmf_event_guard_20260818.tgz \
  jsstrobel@ap2001.chtc.wisc.edu:~/paper_v_cwrmf_event_guard_20260818.tgz
```

## Remote extract + submit (on ap2001)

```bash
cd Holstein_phase3_optuna_chtc
tar -xzf ~/paper_v_cwrmf_event_guard_20260818.tgz
chmod +x chtc/paper_v_cwrmf_event_guard_20260818/*.sh
condor_submit -dry-run /tmp/paper_v_dryrun.ad \
  chtc/paper_v_cwrmf_event_guard_20260818/submit_cwrmf_event_guard_t05_v1.sub
condor_submit chtc/paper_v_cwrmf_event_guard_20260818/submit_cwrmf_event_guard_t05_v1.sub
```

Monitor with `condor_watch_q` on the batch's Condor log file; do not
poll `condor_q` in a loop.

## Later runs

The submit file's `arguments` line is
`<run_id> <final_time> <time_step> <event_gram_hard_floor>`; a full-pulse
`t=4` run is the same lane with `t4_balanced_hardfloor2em4 4.0 0.0025 2e-4`
and a raised `+MaxRuntime`. Do not change profile, cutoff, coupling, or
drive settings in this lane without a new dated submit file.
