# Paper-I local/CHTC numerical parity and Append/RA comparison policy

Date: 2026-08-14

## Bottom line

The recent local RA-ADAPT diagnostic and the original CHTC RA-ADAPT run do
not have numerical-runtime parity. They use the same sealed scientific source,
but they realize that source with different Python/numerical environments and
different hardware/runtime policies. Their accepted-generator paths therefore
cannot be expected to agree bit for bit when symmetry-related candidates are
separated only by floating-point roundoff.

This route difference need not remain scientifically important if a complete,
unmodified local calculation produces a better result. A better local result
can supersede the need to reproduce the old CHTC branch for the purpose of
selecting the best RA-ADAPT trajectory. It does not, by itself, establish a
fair RA-ADAPT versus Append-ADAPT comparison: that comparison still requires
both methods to use the same controlled numerical stack.

This distinction is especially important for the faster local implementation
now under development. Requiring that implementation to reproduce the old
CHTC trajectory bit for bit would defeat its purpose and may be impossible
because the optimized execution route changes numerical evaluation order. The
faster implementation should preserve the scientific algorithm, not the
historical floating-point branch. Its paper-facing Append and RA calculations
should both use the new implementation and one shared, recorded runtime.

## Observed local-versus-CHTC difference

The original Paper-I global-singleton RA-ADAPT calculation used:

- CHTC cluster `9605157`;
- the sealed source archive with SHA-256
  `690d54dbf5bafcaaf974dc11339ed927cb7f5d117265ed51adbb811785740762`;
- the pinned Linux Apptainer image with SHA-256
  `fa5c4ea89a19ad2fa4c264f3bce65d9e83b86b91618ed97dfc80ee7be401239f`;
- `apptainer exec --cleanenv`;
- four requested CPUs per worker;
- no GPU request or CUDA execution path; and
- a 50-round execution horizon.

The local diagnostic used the same sealed source archive, but invoked the
private execution seam directly on macOS ARM64 with the host Python, NumPy,
and SciPy stack. It also wrapped SciPy for telemetry and stopped at round 20.
It did not enter the pinned Linux container or the normal authorization and
worker wrapper.

The first accepted-generator difference appears at round 3. At that point the
CHTC calculation resolved a symmetry-degenerate Phase-0 cluster by a one-ULP
gradient split, while the local calculation produced an exact floating-point
tie and therefore used the pool-index tiebreak. The selected generators were:

- CHTC: pool index 96, `guarded_singleton::eyeeeeze`;
- local: pool index 92, `guarded_singleton::eeeyeeez`.

The underlying Phase-0 magnitudes differed by only about
`5.55e-17`. The two runs had already returned different representatives of a
flat Powell minimum at round 1: their first fitted parameters differed by
about `4.64e-9`, even though their energies differed by only about `2.64e-16`.
Those coordinate-level differences were sufficient to choose opposite sides
of the later symmetry tie.

The 20-round versus 50-round stopping horizon is not the established cause of
the round-3 split. In the sealed implementation, the horizon controls stopping
and observation, while the recorded remaining-evaluation quantity was not
consumed by the relevant Phase-II/III score. The established difference is the
runtime/invocation environment.

## Historical Append-ADAPT versus RA-ADAPT runtime audit

The historical CHTC Append-ADAPT and RA-ADAPT calculations have stronger
parity than the local/CHTC diagnostic, but their effective numerical runtimes
were not fully identical.

| Property | Append-ADAPT | RA-ADAPT |
|---|---:|---:|
| Container image SHA-256 | `fa5c4ea8...401239f` | `fa5c4ea8...401239f` |
| Container entry | `apptainer exec --cleanenv` | `apptainer exec --cleanenv` |
| Condor `getenv` | `False` | `False` |
| Requested CPUs | 1 | 4 |
| Requested GPUs | 0 | 0 |
| Explicit BLAS/OpenMP thread pins | absent | absent |
| Recorded Python/NumPy/SciPy/BLAS/CPU fingerprint | absent | absent |

The byte-identical SIF strongly supports identical image-contained library
files. No GPU was involved. The unresolved variables are the requested CPU
allocation, the realized BLAS/OpenMP threading, and the execute-host CPU
model/features. Consequently, the historical comparison has image parity but
does not have a complete observed numerical-runtime parity receipt.

The relevant historical packages are:

- Append-ADAPT:
  `chtc/paper_i_ra_adapt_repair_20260727/`
  `paper_i_append_adapt_stationary_core12_r70_fresh_20260731_v1_chtc`;
- RA-ADAPT:
  `chtc/paper_i_ra_adapt_repair_20260727/`
  `paper_i_ra_adapt_global_singleton_gradient_phase0_phase123_qiskit_`
  `phase23_no_lanes_cap24_tau1em4_r50_20260807_v1_chtc`.

## When a better local result makes old-route reproduction unnecessary

The accepted route is not a unique mathematical observable when several
symmetry-related candidates have equal scores within floating-point
resolution. The scientific quantities of interest are the resulting energy,
accuracy, and resource costs. Therefore, exact reproduction of the old CHTC
branch is unnecessary if a new local trajectory satisfies all of the
following:

1. It uses the same scientific contract: Hamiltonian, cutoff, reference,
   executable pool, Phase-0 through Phase-III definitions, insertion policy,
   optimizer, optimizer budget, and seeds.
2. It is a clean calculation rather than an instrumented telemetry replay:
   no optimizer monkeypatch and no private-seam behavior change.
3. It reaches the complete paper-facing horizon, or the exact prefix required
   by the named comparison. A round-20 diagnostic alone cannot supersede a
   completed round-50 result.
4. Its same-cutoff energy error and resource receipts are complete. “Better”
   means lower error at the relevant prefix, or attainment of the same target
   error with lower `S_alg` and/or compiled resources under the manuscript's
   declared comparison rule.
5. Its source, runtime, checkpoints, accepted-round receipts, and terminal
   result are preserved and validated.

If these conditions hold, the new local trajectory may become the
authoritative RA-ADAPT result and the old CHTC branch may remain as historical
provenance. The numerical bifurcation should then be described as a
symmetry-tie route difference rather than treated as a failed scientific
calculation.

## Faster local implementation: required parity

The faster local implementation has three distinct parity questions:

1. **Scientific-semantics parity is required.** It must evaluate the same
   resolved problem, executable pool, phase scores, insertion policy,
   optimizer protocol, accepted refit, stopping rule, and accounting
   definitions. A speedup may change data structures, caching, batching of
   independent calculations, compilation strategy, or evaluation order only
   when those changes preserve this contract.
2. **Direct-comparison runtime parity is required.** Append-ADAPT and RA-ADAPT
   results compared against each other must both be produced by the faster
   implementation under the same pinned local numerical runtime. Comparing
   faster-local RA against old-CHTC Append would confound method and runtime.
3. **Historical-route parity is not required.** The accepted generator at an
   exact or near-exact symmetry tie may differ from the old CHTC choice. The
   old ordered generator sequence is not itself the scientific target.

The faster implementation should therefore be validated by a differential
semantic test rather than an exact old-route replay. At fixed accepted states
and candidate inputs, the old and new implementations should agree on
Hamiltonian energies, gradients, phase scores, admissible candidate sets,
insertion positions, and accounting receipts within named numerical
tolerances. Exact identity should still be required for discrete scientific
objects such as the problem, pool, guards, policies, seeds, and receipt
schemas. When a symmetry-degenerate score cluster falls within the declared
tolerance, either branch is admissible if the implementation applies its
documented deterministic tiebreak.

Before promotion, the faster route should provide:

- a focused old-versus-new semantic differential suite;
- a clean full-horizon local RA run;
- a matched full-horizon local Append run using the same faster stack;
- complete checkpoints and accepted-round, cost, and runtime receipts; and
- an objective comparison showing whether the new RA trajectory improves
  energy error, matched-accuracy cost, or both.

This validation supports a new authoritative local evidence set. It does not
retroactively turn the older CHTC sequence into a parity replay, and it does
not require slowing the new implementation down to imitate the historical
execution order.

## What still requires matched runtime parity

Any direct RA-ADAPT versus Append-ADAPT claim must compare results produced
under one shared numerical-runtime contract. A better RA trajectory does not
remove this requirement because runtime differences could otherwise favor one
method's optimizer path or tiebreaks.

The next paper-facing paired calculation should require:

- the same verified SIF bytes;
- the same requested CPU count;
- CPU-only execution;
- `OMP_NUM_THREADS=1`;
- `OPENBLAS_NUM_THREADS=1`;
- `MKL_NUM_THREADS=1`;
- `VECLIB_MAXIMUM_THREADS=1`;
- `NUMEXPR_NUM_THREADS=1`;
- `BLIS_NUM_THREADS=1`;
- `OMP_DYNAMIC=FALSE` and `MKL_DYNAMIC=FALSE`;
- `PYTHONHASHSEED=0`;
- recorded Python, NumPy, SciPy, Qiskit, NumPy configuration, libc, CPU model,
  CPU feature, affinity, and loaded BLAS/LAPACK identities; and
- a fail-closed comparison gate that admits the pair only when the observed
  Append and RA runtime fingerprints match exactly.

Running a matched Append/RA pair sequentially in the same Condor worker and
container is the strongest practical way to close host and library-dispatch
differences. Separate workers are admissible only when their complete runtime
fingerprints agree.

## Implemented guard and remaining integration

The repository now contains the numerical-runtime contract and receipt logic
in:

- `pipelines/static_adapt/ra_adapt/numerical_runtime.py`.

The matched Study-1 bundle materializer now requires one shared contract and
copies it into every execution template:

- `pipelines/static_adapt/ra_adapt/bundles.py`.

Focused coverage is in:

- `test/test_ra_adapt_numerical_runtime.py`;
- `test/test_ra_adapt_bundles.py`.

The verified command was:

```text
PYTHONPATH=. pytest -q -p no:cacheprovider \
  test/test_ra_adapt_numerical_runtime.py \
  test/test_ra_adapt_bundles.py
```

Result: `43 passed`.

This is not yet an end-to-end gate for the current page-facing CHTC package
builders. Those workers do not yet produce the runtime receipt, and the report
builders do not yet require the pair-level parity receipt. No CHTC rerun was
submitted as part of this audit. A new paired package must bind the verified
image digest and runtime receipt into each result before the paper-facing
Append/RA comparison is parity-certified.

## Decision rule

| Goal | Required action |
|---|---|
| Reproduce the old CHTC generator sequence | Run the sealed source through the same pinned CHTC numerical environment. |
| Validate the faster local implementation | Prove scientific-semantic equivalence with differential tests; exact historical route identity is optional at numerical ties. |
| Obtain the best admissible RA-ADAPT trajectory | A complete, clean, objectively better local trajectory may supersede the old branch. |
| Compare RA-ADAPT with Append-ADAPT | Run both with the faster implementation under the same complete local numerical-runtime contract. |
