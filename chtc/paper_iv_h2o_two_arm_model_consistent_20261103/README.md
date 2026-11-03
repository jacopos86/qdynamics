# paper_iv_h2o_two_arm_model_consistent_20261103 — campaign contract

Non-submitting CHTC package for the Paper-IV H2O static construction on the
model decided by the user on 2026-11-02: **18 qubits, sector dimension
14,400, work cutoffs (3,3,3)**, run as bounded segments that exit normally so
every segment's results transfer. Nothing in this directory submits, opens
SSH, or changes remote state.

It is the `paper_iv_h2o_two_arm_20260903` package with the fixture, the
campaign name, the qubit count, the memory request and a fixture-identity
guard changed; the segment discipline, the two arms, the runner, the
segment chaining and the monitor are the same code. Read that package's
README for the reasoning behind the segment arithmetic; this file states what
is different and what the new model is.

Handoff authority: `agent_guidance/shared/paper-iv-h2o-chtc-handoff-20260903.md`
(§0, §3, §4, §5) and the worklog entry "Production fixture from the vibronic
builder — 2026-11-03" in `agent_guidance/shared/paper-iv-worklog.md`.

## The model this campaign runs

Two decisions on 2026-11-02 (scientific record:
`work/doron-chemistry-brief-20261102.md` in the main checkout):

1. Oscillator frequencies `omega_nu^2 = K_nu^PES + chi_nu`: the frozen-core FCI
   surface curvature along the mode plus the model's own correlated response to
   the assembled coupling, so the exactly solved model carries the FCI
   curvature. The RHF Hessian of the previous model double counts the orbital
   relaxation the coupling regenerates (its symmetric-stretch curvature is about
   a third of the FCI surface's, which is why the previous campaign needed
   `n_max = 7` on that mode).
2. Model-consistent modes: the eigenvectors of `K_PES + chi` as matrices, so the
   whole FCI Hessian is reproduced, not only its diagonal (bend and symmetric
   stretch mix by 2.6 degrees). Displaced orbitals are aligned block by block so
   the frozen core stays canonical.

Produced by `src/quantum/chemistry/paper_iv_h2o_fixture_from_builder.py` from
the molecule-agnostic builder through the lane's own generator, so derivative
validation, operator encoding, the exact reference, the evidence gate and the
byte-sha attestation are unchanged.

| quantity | this campaign | `paper_iv_h2o_two_arm_20260903` |
|---|---|---|
| fixture | `h2o_builder_model_consistent_work333_ref444_20261103_fixture.json` | `h2o_linear_fd_work373_ref484_fixture_20260826.json` |
| sha256 of the bytes (attested by every checkpoint) | `de32cd4d1e29b8dcc07e94ad132ee43893fe03fd36cf4538eb6dadf6d5e194df`, 15,673,312 bytes | `190387449b…256e`, 28,025,185 bytes |
| qubits / sector dimension | 18 / 14,400 | 19 / 28,800 |
| work / reference cutoffs | (3,3,3) / (4,4,4) | (3,7,3) / (4,8,4) |
| modes bend / symmetric / antisymmetric (cm⁻¹) | 2130.79 / 5314.42 / 5061.82 | 2170.79 / 4138.69 / 4389.24 |
| exact ground energy (the ED target) | −74.99991178 Ha | −75.01446351 Ha |
| ground-state occupations at the work cutoffs | 0.05 / 0.20 / 0.01 | up to 2 quanta on the symmetric stretch |
| `STATIC_ADAPT_PAULI_SIGN_CACHE_MAX_NQ` | 18 | 19 |
| `request_memory` | 16GB (see below) | 8GB in the package; the live campaign was raised to 22,528 MB |

`build_package.py` refuses any other fixture: it checks the byte count, the
sha256, `register_layout.n_total_qubits == 18`, `physical_sector.n_ph_max_by_mode
== [3,3,3]`, the lane mode order, `production_validated`, and that the fixture's
`provenance.backend.model_definition` carries `decided_model: true` with
`pes_consistent` / `block_diagonal` / `model_consistent`. The identity record
goes into `package/input_manifest.json` and `runtime_preflight.py` re-verifies
it against the packaged file.

The registry checks the request's scalar `n_ph_max` against the fixture's
largest cutoff: `h2o_application` reads it from the fixture, so nothing in the
runner changes for (3,3,3), but any hand-written request for this fixture
must carry `n_ph_max = 3` (the old campaign's was 7).

## The two arms

| Arm | `--candidate-representation` | JobBatchName |
|---|---|---|
| `macro` | `macro_generator_v1` | `paper_iv_h2o_two_arm_model_consistent_20261103_macro` |
| `singleton` | `sector_complete_pauli_block_v1` | `paper_iv_h2o_two_arm_model_consistent_20261103_singleton` |

Both arms, every segment: `--insertion append_only`,
`--checkpoint-history-tail 1`, `--maximum-controller-rounds 50` (the campaign
horizon), estimator ledger enabled, `--cost-mode proxy` by default.
`marrakesh_graph_span_v1` needs an **18-qubit** FakeMarrakesh embedding: the
19-qubit boson-block spans chosen for the previous campaign (handoff §0.1,
lane (10,8,8) / B (8,7,6) / F (1,2,1) for registers of 2+3+2 qubits) do not
apply to registers of 2+2+2 qubits, so graph-span is a new user decision here;
`validate_cost_mode_register_size` refuses it before any file is written.

## Segment discipline

Unchanged from the previous package: pool wall 259,200 s (hold, no
stage-out); `periodic_hold` 172,800 s in every submit file; in-process
`--max-wall-seconds` 129,600 s by default. The 36 h default was sized for the
19-qubit model; this model's rounds are cheaper (half the sector dimension), so
the default leaves more margin, not less. Rebuild with `--max-wall-seconds`
once per-round seconds are measured for this model.

Requests: `request_cpus = 4`, `request_disk = 10GB`, `request_memory = 16GB`.
The previous campaign measured a true peak of 14,792 MB at round 25 for the
19-qubit (3,7,3) model (sector 28,800) and was raised to 22,528 MB; this model
has half that sector and 18 qubits, and 16 GB is that peak scaled by the
sector ratio with a factor-of-two margin. No peak has been measured for this
model: `build_package.py --request-memory 22528MB` reproduces the previous
cap if the first segment shows the estimate was wrong (a memory hold never
stages out, so err high on the first segment and read its receipt).

## What transfers, when, what is user-gated, files, workflow

Identical to `paper_iv_h2o_two_arm_20260903/README.md` with the campaign name
replaced: results land under
`/home/jsstrobel/chtc/paper_iv_h2o_two_arm_model_consistent_20261103/<arm>/segNN/results/`
(`--results-mode home`), the user does every external-state change
(`ssh -MNf ap2001`, the rsync of this directory, `condor_submit`, any release,
removal or deletion, requesting `/staging`, adopting evidence), and the files
play the same roles (`build_package.py`, `run_segment.sh`, `next_segment.py`,
`monitor.sh`, `runtime_preflight.py`).

```bash
PY=/Users/lordskyler/Holstein_test/.venv/bin/python
cd /Users/lordskyler/Holstein_test.worktrees/claude-paper-iv-fixture-from-builder-20261102
$PY chtc/paper_iv_h2o_two_arm_model_consistent_20261103/build_package.py       # writes package/, submit_*.sub; refuses any other fixture
$PY chtc/paper_iv_h2o_two_arm_model_consistent_20261103/runtime_preflight.py   # must print "status": "pass"
# --- user: ssh -MNf ap2001; rsync the directory; condor_submit submit_macro.sub / submit_singleton.sub
bash chtc/paper_iv_h2o_two_arm_model_consistent_20261103/monitor.sh <cluster>,<cluster>
$PY chtc/paper_iv_h2o_two_arm_model_consistent_20261103/next_segment.py --previous-results <mirror>/macro/seg00/results
```

`git archive` ships HEAD: the source that runs on the pool is the commit of
this branch recorded in `package/input_manifest.json` (`source_archive.git_head`).
That commit carries the vibronic builder and the fixture producer, but the
engine, the registry and `h2o_application` are the live refit branch's
(`d17d4046`) unchanged.

## Status of the previous package's blockers

1. `--resume-checkpoint` for an H2O checkpoint: **admitted** since `73854cbc`
   (named `paper_iv_h2o_linear_fd_application` resume branch with fixture-sha
   attestation); the live campaign has resumed thirteen segments on it. The
   attestation also means a checkpoint from the previous model's fixture can
   never be resumed on this one, by design.
2. Finalization crash after the horizon: fixed at `929f1b47` (the archived
   `runtime.py` carries the H2O skip; the preflight checks it).

## Before the first submission (user)

- Decide the fate of the previous campaign's running segment (cluster
  `12396953.0`, seg13 of `paper_iv_h2o_macro_refitfix_20260911`, old model):
  this package neither stops nor replaces it.
- Confirm `image.sif` still exists on ap2001 and `quota -s` shows headroom
  (the package is ~22 MB: 15.7 MB fixture plus the source archive).
- Point the monitor and any restart watcher at this campaign's batch names.
