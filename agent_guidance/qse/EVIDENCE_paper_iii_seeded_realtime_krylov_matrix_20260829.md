# Paper-III seeded real-time-Krylov comparator evidence — 2026-08-29

## Verdict

The fixed-policy, single-seed real-time-Krylov comparator matrix is complete as
candidate evidence, but the method is **not** a complete six-root comparator
for every Hubbard--Holstein regime.  With one-to-one root assignment, 10 of 16
cases reach the route-local `1e-6` target and 6 do not.

The six failures are scientifically meaningful.  The former diagnostic matched
each exact target to its nearest pencil root independently, so one Ritz root
could be reused for multiple degenerate targets.  That convention produced six
false six-root successes.  The admitted audit now uses a Hungarian one-to-one
assignment and checks target-subspace overlaps; it does not permit root reuse.

## Locked route

- Method: seeded real-time Krylov, one deterministic complex sector kick.
- Policy: `seeded_realtime_krylov_fixed_dt_0p5_v1`.
- Fixed time step: \(\Delta t=0.5\) for every case; no case-local tuning.
- Seed: `20260826`, projected orthogonal to the prepared ground state.
- Supported-overlap cutoff: `1e-12` relative to the largest overlap eigenvalue.
- Hubbard--Holstein cap: `k=100`; Peierls--Hubbard cap: `k=40`; SIAM cap:
  `k=100`.
- Exact excited roots and vectors are inaccessible to the online route.  They
  enter only after termination for one-to-one energy matching, spectral-window
  gaps, overlaps, residuals, and transition-spectrum audits.
- Cost \(K\): aggregate graph-span two-qubit count for preparing every Krylov
  basis state, charging `k(k-1)/2` Hamiltonian steps.
- Qiskit cost: the same aggregate step count using FakeMarrakesh backend
  transpilation (`seed_transpiler=7`, optimization level 1).
- Logical estimator queries: all independent overlap and Hamiltonian pencil
  entries with no QWC reuse.
- Source-preparation cost is excluded, matching the inherited comparator
  convention; the artifact states this explicitly.

## Route-local `1e-6` results

For successful cases the row is the first one-to-one six-root crossing.  For a
non-reaching case it is the `k=100` terminal row, so the displayed error and
resources are achieved terminal values rather than a fictitious crossing.

| Case | Status | k | rank | K | Qiskit N2q | Qiskit D2q | estimator queries | max six-root error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HH intermediate-strong, nph1 | NOT REACHED | 100 | 14 | 99,000 | 143,550 | 118,800 | 90,900 | 1.914e-1 |
| HH intermediate-strong, nph3 | REACHED | 57 | 48 | 108,528 | 213,864 | 108,528 | 52,896 | 6.161e-7 |
| HH intermediate-weak, nph1 | NOT REACHED | 100 | 14 | 99,000 | 143,550 | 118,800 | 90,900 | 2.360e-1 |
| HH intermediate-weak, nph3 | REACHED | 58 | 45 | 112,404 | 221,502 | 112,404 | 54,752 | 7.072e-7 |
| HH strong-strong-u8, nph1 | NOT REACHED | 100 | 14 | 99,000 | 143,550 | 118,800 | 90,900 | 5.187e-1 |
| HH strong-strong-u8, nph3 | NOT REACHED | 100 | 54 | 336,600 | 663,300 | 336,600 | 161,600 | 4.845e-1 |
| HH strong-weak-u8, nph1 | NOT REACHED | 100 | 14 | 99,000 | 143,550 | 118,800 | 90,900 | 5.261e-1 |
| HH strong-weak-u8, nph3 | NOT REACHED | 100 | 54 | 336,600 | 663,300 | 336,600 | 161,600 | 5.200e-1 |
| HH weak-strong, nph1 | REACHED | 19 | 14 | 3,420 | 4,959 | 4,104 | 3,420 | 5.304e-8 |
| HH weak-strong, nph3 | REACHED | 56 | 47 | 104,720 | 206,360 | 104,720 | 51,072 | 4.848e-7 |
| HH weak-weak, nph1 | REACHED | 19 | 14 | 3,420 | 4,959 | 4,104 | 3,420 | 4.567e-7 |
| HH weak-weak, nph3 | REACHED | 60 | 45 | 120,360 | 237,180 | 120,360 | 58,560 | 3.275e-7 |
| Peierls u8-weak | REACHED | 15 | 15 | 10,500 | 14,595 | 13,125 | 3,600 | 4.756e-8 |
| Peierls weak-strong | REACHED | 24 | 15 | 27,600 | 38,364 | 34,500 | 9,000 | 1.031e-10 |
| Peierls weak-weak | REACHED | 21 | 15 | 21,000 | 29,190 | 26,250 | 6,930 | 1.131e-9 |
| SIAM \(N_b=4\) | REACHED | 86 | 62 | 124,270 | 124,270 | 80,410 | 67,338 | 3.162e-7 |

## Static spectral audits

Every case has a terminal post-run audit containing:

- all six matched root energies and errors;
- the sixth root and the sixth-to-seventh-root gap;
- the one-to-one root-overlap matrix and target-subspace principal cosines;
- six terminal Ritz residual norms;
- two declared static observables with exact and approximate transition
  strengths and Lorentzian-broadened spectral errors.

The subspace audit is load-bearing.  It exposed that the old independent-nearest
matching convention could report small energy error even when the six-vector
target window was rank-deficient at a degeneracy.

## Provenance and validation

- Implementation commit tested by the artifact: `6a5061f8`.
- Full command:
  `python3 pipelines/exact_bench/paper_iii_seeded_realtime_krylov_campaign.py`
- Guarded local peak: 290 MB under a 4 GB process-tree limit.
- Focused and neighboring tests: `26 passed`.
- Machine-readable artifact:
  `output/diagnostics/paper_iii_seeded_realtime_krylov_matrix_20260829_v1/matrix.json`
- Artifact SHA-256:
  `efccf7a58aca4b44ee4da9f8853306a559d3df6d3404898689bd1d70340af98c`.
- Receipt:
  `output/diagnostics/paper_iii_seeded_realtime_krylov_matrix_20260829_v1/receipt.json`.
- Hamiltonian-power QKSD status: `IMPLEMENTATION_PENDING`; no Davidson or other
  method was substituted.

## Next algorithmic decision

Do not describe the single-seed chain as a universally complete six-root
comparator.  If a Krylov route that resolves degenerate multiplicity is needed,
the next controlled experiment is a separately named **block-seeded real-time
Krylov** comparator.  Its multiple independent seed preparations and all block
pencil entries must be charged.  That is a new comparator variant, not an
automatic fallback or a silent repair of this evidence.
