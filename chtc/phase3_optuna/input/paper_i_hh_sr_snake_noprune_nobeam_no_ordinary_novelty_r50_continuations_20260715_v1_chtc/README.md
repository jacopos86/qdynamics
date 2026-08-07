# SR-SNAKE round-30 to round-50 continuations

Status: submitted to CHTC as cluster `8811168` (four jobs). The builder itself
only stages and validates the bundle; the authenticated submission and initial
scheduler acceptance are recorded in `submission_record.json`.

The four intended rows are `strong_weak_u8`, `weak_strong`,
`intermediate_strong`, and `strong_strong_u8`. Each row must continue its own
completed, pre-terminal round-30 `current.json` checkpoint and the exact sibling
`estimator_call_ledger.json`. It also extracts the exact round-30
`post_admission_prune` signed-prefix checkpoint into the compact canonical
`signed_active_prefix_checkpoint.json` sidecar. That same sidecar carries the
single unique round-30 maturity-controller snapshot and its canonical sorted
JSON digest, so Phase-I/II/III live/null streak state is resumed rather than
reinitialized. The full source result remains local and is authenticated by
path and SHA-256; it is not transferred. The
source checkpoint and large source ledger are transferred as deterministic gzip
(`mtime=0`) and are decompressed back to the exact canonical sibling filenames
only after both compressed and original uncompressed hashes and sizes pass. The builder fails closed
when any source artifact needed to authenticate the continuation is missing.

The scientific source is the preserved archive with SHA-256
`94c2df6df22c6d277aefdd6559273d943e3724d476ecab6648c6dd11e1fd78c6`, plus
the verified, manifest-locked no-beam continuation patch under `source_lock/`.
The builder never imports the live scientific tree into `source_locked.tar.gz`.

Build the complete four-row bundle only after all four fetched sources exist:

```bash
python3 chtc/phase3_optuna/input/paper_i_hh_sr_snake_noprune_nobeam_no_ordinary_novelty_r50_continuations_20260715_v1_chtc/build_bundle.py
```

For preparation/testing before all sources are fetched, stage only complete
rows and record every other row as blocked:

```bash
python3 chtc/phase3_optuna/input/paper_i_hh_sr_snake_noprune_nobeam_no_ordinary_novelty_r50_continuations_20260715_v1_chtc/build_bundle.py --ready-only
```

The continuation contract changes only the cumulative horizon from 30 to 50,
adds the typed resume/segment controls, and redirects operational output/cache
paths. It performs at most 20 new singleton admissions, requires a
`FakeMarrakesh` prefix compile smoke, restores the state-keyed estimator ledger,
and skips a boundary refit only after exact checkpoint verification.

Resource contract:

- all rows: 4 CPUs, 61,440 MB disk, `MaxRuntime=259200` seconds;
- `strong_weak_u8`: 32,768 MB memory;
- the three strong-phonon rows: 40,960 MB memory.

Any future resubmission remains a separate, explicitly authorized action.
Before resubmitting, verify the remote execution-image hash, run the recorded
Condor preflight, and require `preflight.json` to contain four ready rows.
