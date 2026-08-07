# Weak-strong NPH7 SR outer-information matched pair

This directory is a source-locked, two-node CHTC bundle for one matched
weak-strong Hubbard--Holstein comparison at `n_ph_work=n_ph_reference=7` and
controller round 50.

The DAG order is immutable:

1. `CONTROL` runs the authoritative current-runtime SR-SNAKE job.
2. A DAGMan POST script validates the complete result, exact query closure,
   Qiskit sidecar, checkpoint, source/image hashes, and writes a dynamic control
   gate.
3. `REUSE` runs only after that gate exists and differs scientifically only by
   `--adapt-formal-manifold-route-profile
   sr_no_prune_symmetric_cost_outer_information_active_v1`.

Both nodes retain the source-locked SR selector, supported-FS Powell refit,
fallback-only novelty, no beam, no batching, no pruning, no ordinary novelty
multiplier, seed 7, and separate cold caches.  Neither node shares mutable
optimizer or measurement-cache state.

## Eviction-safe continuation

Each node uses HTCondor self-checkpoint exit code 85.  On eviction the wrapper
stops the complete Apptainer process group, atomically packages the latest
finalized `current.json` together with the exact content-addressed estimator
ledger checkpoint authenticated by that current, and transfers only the
mode-local checkpoint archive.  Restart validates the source lock, job hash,
scientific-command digest, route/frame/ledger provenance, and cumulative outer
round before restoring into a mode-private resume-input directory.  It then
clears only that node's ephemeral output/cache root and appends the frozen
structural-resume flags to the otherwise byte-identical command.  Repeated
eviction before another completed round preserves the prior checkpoint; an
eviction before round one uses an authenticated cold-start sentinel.  CONTROL
and REUSE checkpoints cannot cross modes.

## Source-lock and execution gates

The frozen v5r2 runtime includes narrow, regression-tested exact-metric
fallbacks for both an uncertified novelty-fallback transition and the canonical
`eps_grad_suppressed_continue` singleton path.  In either case it preserves the
authoritative SR admission, invalidates only transported geometry, charges the
exact Powell metric, and requires later exact reanchoring.  The source audit
records 80 passing focused tests and the strong-weak live repair boundary was
observed through round 33; that observation is nonterminal repair evidence, not
a completed scientific result.

`submission_gate.json` permits worker startup only for the exact v5r2 archive
and audit.  CHTC quota, image, and Qiskit/FakeMarrakesh checks remain required at
the later submission boundary.  Structural rollback remains forbidden.

## Build and verify

From the active local checkout:

```bash
python3 chtc/phase3_optuna/input/paper_i_hh_sr_outer_information_v5r2_weak_strong_nph7_r50_20260719_v1_chtc/build_bundle.py
PYTHONPATH=chtc/phase3_optuna/input/paper_i_hh_sr_outer_information_v5r2_weak_strong_nph7_r50_20260719_v1_chtc \
  pytest -q chtc/phase3_optuna/input/paper_i_hh_sr_outer_information_v5r2_weak_strong_nph7_r50_20260719_v1_chtc/test_bundle.py
```

No authentication, upload, or submission is performed by this bundle build.
Before a later submission, rerun local tests, the exact
Apptainer/Qiskit/FakeMarrakesh preflight, and CHTC quota/image checks.  A valid
submission must start with no stale `anchor_gate.control.json`; the POST gate
creates it only from the current CONTROL transfer.
