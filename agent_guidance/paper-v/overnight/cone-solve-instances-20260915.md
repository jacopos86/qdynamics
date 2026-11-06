# Conic solve instances (key `cone-solve-instances`), 2026-09-15

Worktree `/Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914`,
branch `claude/paper-v-routes-20260914`, round 2 of the overnight loop.
Round-1 finding this attacks: over `t=0` to `0.01` at `h=0.0025` in the strict
always-on profile the rollout spends `142.350 s` of `145.172 s` inside
Clarabel over 36 calls (98.1 percent); after the four bitwise-neutral
accelerations the same rollout is `139.883 s` with `138.644 s` in the solve.
Nothing is committed; strict and balanced defaults are unchanged.

Repair round (2026-09-15, after an independent review of the first
submission).  Four corrections are folded in below and listed here so the
difference from the reviewed text is visible: the warm-started SCS
acceptance claim is restated as measured and its polish gate is repaired in
code and re-measured (section (b), group (5)); the recorded interior-point
iteration range is `14` to `28`, not `17` to `28`; the recording rerun's
extra Clarabel-tier seconds are contention inside the solver, not the
instance write, which lands outside the timed region; and the rows that said
`ram_guard peak` are named for the field they came from, the driver
process's own `integration.peak_resident_bytes`.  No default, acceptance
rule or profile changed in the repair.

## Negative result first

Of the 22 declared candidates replayed on the 36 recorded conic instances,
none that passes the strict full-Gram acceptance test on all 36 is faster
than the pinned single-threaded Clarabel call by more than the machine's own
spread.  A second bench pass brackets that spread at `0.077 s` on a
`135.7 s` baseline and shows the best strict-passing candidate,
`presolve_enable = False`, at `135.711 s` -- exactly free, exactly worthless
for time, and bitwise identical on all 36 solutions.  Loosening the
tolerances one and two decades buys `1.06x` and `1.11x`; a warm-started SCS
is `5.2x` slower and its answer passes the substituted-inflation balanced
acceptance test on 15 of the 36 against the recorded solution's 29 (20 of 36
verdict agreement); a reduced bundle is up to `46.6x` faster and satisfies
the strict acceptance test on none of the 36 at any threshold tried.  The one real
speedup is multithreaded factorization: `max_threads = 8` is `1.64x` on the
instances and runs the strict `t=0` to `0.01` rollout in `92.418 s` against
the anchor's `139.883 s` (`1.51x`), with the 31 reported archive coordinates
agreeing to `2.088607e-15` -- but it fails the strict acceptance test on one
of the seven instances the pinned run accepts, so it is the balanced-passing
profile, not the strict one, and it is off by default.

Impact analysis, read-only from the main checkout's index
(`node /Users/lordskyler/Holstein_test/.gitnexus/run.cjs impact <name>
--direction upstream --repo /Users/lordskyler/Holstein_test`; the index is 8
commits behind this worktree, so every verdict was confirmed by grep):

| symbol | risk | epistemic | upstream callers |
|---|---|---|---|
| `_solve_relaxed_readable_bundle_cone` | LOW | exact | `_solve_relaxed_readable` -> `radial_atom` -> `integrate_cwrmf_ssprk2` |
| `_solve_critical_schur` | LOW | exact | `radial_atom` -> `integrate_cwrmf_ssprk2` -> `apcm_carried_witness_analysis.main` |
| `_polish_terminal_completion_velocity` | LOW | exact | `_solve_relaxed_readable_bundle_cone` -> `_solve_relaxed_readable` -> `radial_atom` |
| `_solve_relaxed_readable_full_cone` | LOW | exact | `_solve_relaxed_readable` -> `radial_atom` -> `integrate_cwrmf_ssprk2` |
| `radial_atom` | LOW | exact | `integrate_cwrmf_ssprk2` -> `apcm_carried_witness_analysis.main` |
| `CWRMFSettings` | LOW | exact | `apcm_carried_witness_analysis`, `apcm_carried_witness_mode_audit`, `apcm_carried_witness_campaign`, `apcm_carried_witness_stitch`, `stability/__init__` |

`grep -rn "CWRMFSettings(" paper_5` finds six construction sites, all
keyword-only, so a new field with a default is additive.

Repair round, same command on the three symbols it touches:
`WarmStartedScsCandidate` and `declared_candidates` return
`"error": "Target not found"` with `risk: UNKNOWN`, and `acceptance`
returns `impactedCount: 0` with `risk: UNKNOWN` and the riskNote that an
empty caller set is not evidence of disuse -- all three are expected, since
the modules are new and untracked and the index is on the main checkout.
Confirmed by text search instead:
`grep -rn --include='*.py' -e WarmStartedScsCandidate -e declared_candidates
-e polish_profile -e BALANCED_INFLATION_SHIFT paper_5 pipelines` finds them
only in `cwrmf_solver_replay.py`, `cwrmf_solver_replay_bench.py` and
`cwrmf_solver_replay_test.py`, all three owned here, and nothing on the
guarded rollout path reaches them.

## (a) Conic instance recorder

New module `paper_5/src/paper5/stability/cwrmf_solver_instances.py`.
`ConicInstanceRecorder` writes, for every Clarabel call the guard makes, the
objective matrix `P` and vector `q`, the cone matrix `A` and right-hand side
`b`, the cone specification as `(constructor name, dimension)` in order, the
Clarabel settings the call site set, the tier name, the radial atom index and
its time, the within-tier iteration, the step size, and the solution Clarabel
returned (`x`, `status`, `obj_val`, `r_prim`, `r_dual`, `iterations`,
`solve_time`).  It also stores, per instance, what is needed to rebuild the
62-row Gram of any candidate: the base velocity `x` adds to, the completion
indices it adds on, the shifted 62-row Gram of the unconstrained endpoint,
the desired completion velocity, the completion radius squared, the retained
scaled norm and the velocity ceiling; and once per run, the step-scaled
`(340, 62, 62)` completion coefficient tensor and the 231 readable completion
indices.

Wired at the four Clarabel call sites in
`paper_5/src/paper5/stability/apcm_carried_witness.py` -- the critical Schur
tier, the adaptive bundle tier, the terminal completion polish, the full
62-row cone -- plus one `begin_atom` call at the top of `radial_atom`.  Off
by default: `CarriedWitnessModel._instance_recorder` is `None` and an
uninstrumented rollout pays one attribute read per solver-tier iteration.
Driver flag `--record-conic-instances`
(`apcm_carried_witness_analysis.py`), which writes under
`conic_instances/` in the run directory and reports
`summary["integration"]["recorded_conic_instances"]`.

### Parity

Rerun of the strict always-on profile over `t=0` to `0.01` at `h=0.0025`
(4 SSPRK2 steps, 8 radial atoms, strong coupling `lambda=1.5`, `gamma=0.5`,
drive amplitude 1, phonon cutoff 16, exact-correlated preparation) with
recording on.  Artifact:
`output/local_runs/paper_v_cwrmf_record_strict_t001_h0025_20260915_v1/`.

Every array of `autonomous_trajectory.npz` and `trajectory.npz` except
`maximum_atom_seconds` is equal under `np.array_equal` to the round-1
post-change anchor
`output/local_runs/paper_v_cwrmf_final_strict_t001_h0025_20260914_v1`; worst
relative difference `0.000e+00` over all 18 arrays.  The all-coordinate
scalar RMS against the exact cutoff-16 reference is `5.827261740618364e-05`
in both, to every printed digit.

| quantity | anchor (round 1) | recording rerun |
|---|---:|---:|
| autonomous wall seconds | `139.883` | `148.272` |
| Clarabel solve tier seconds / calls | `138.644` / 36 | `146.807` / 36 |
| coefficient assembly and svec packing seconds | `0.571` | `0.610` |
| 62-row eigendecomposition seconds | `0.047` | `0.047` |
| driver-process peak resident set, `integration.peak_resident_bytes` (`RUSAGE_SELF`), MiB | `1609` | `1859` |
| recorded conic instances | -- | 36 |

The `8.163 s` difference in the Clarabel tier is not the instance write.
At all four call sites the order is `clock.add("clarabel_solve",
solve_started)` first and `self._instance_recorder.record(...)` second, so
the 242 MB write lands outside the timed region, and the summaries agree:
`unaccounted_seconds` rises only `0.2465` to `0.3983` between the anchor and
the recording rerun.  What rose is the solver's own reported time -- the 36
recorded `solve_time` values sum to `145.819 s`, more than the anchor's
entire `138.644 s` tier -- so the difference is contention on a machine
carrying two other heavy runs and the write-back of the instance files, not
the write inline in the tier.  The trajectory is unaffected, which is the
gate.

### What the 36 instances are

33 are bundle-tier iterations and 3 are critical Schur SQP iterations; 30
returned `Solved` and 6 `AlmostSolved`.  The sum of Clarabel's own reported
`solve_time` over the 36 is `145.819 s`.

| atom | time | tier | iterations | positive-semidefinite cone side dimension | Clarabel solve_time, s |
|---:|---:|---|---:|---|---:|
| 0 | 0.00000 | critical Schur | 3 | 12 | 0.006, 0.006, 0.006 |
| 1 | 0.00250 | bundle | 13 | 12 -> 52 | 0.029 to 1.988, sum 11.24 |
| 2 | 0.00250 | bundle | 6 | 52 -> 62 | 1.770 to 4.420, sum 19.51 |
| 3 | 0.00500 | bundle | 5 | 62 -> 70 | 3.388 to 7.030, sum 26.87 |
| 4 | 0.00500 | bundle | 2 | 70 -> 72 | 5.713, 9.901 |
| 5 | 0.00750 | bundle | 3 | 72 -> 76 | 7.064, 8.969, 11.267 |
| 6 | 0.00750 | bundle | 2 | 76 -> 78 | 8.622, 11.299 |
| 7 | 0.01000 | bundle | 2 | 78 -> 80 | 10.884, 14.462 |

The cone side dimension is twice the spectral bundle rank, because
`_realify_hermitian` doubles the 62-row complex Hermitian block.  Clarabel's
recorded iteration count over the 33 bundle calls runs from 14 to 28 -- 14
once, on the first bundle call of atom 1 at side dimension 12, then 17 to 28
on the remaining 32, with 22 the most frequent value (8 calls) -- while the
solve time rises from `0.029 s` at side dimension 12 to `14.462 s` at side
dimension 80; the three critical-Schur calls took 12 iterations each.  The
cost is the per-iteration factorization of a growing positive-semidefinite
block, not more interior-point iterations.

Two structural facts follow from the record, both visible only now that the
instances are stored:

- Within one atom only the last bundle iteration's solution is used; the
  earlier ones exist to enlarge the basis.  26 of the 33 bundle calls are
  such enlargement iterations, and the bench confirms it independently: the
  strict acceptance test on the full 62-row Gram passes on exactly 7 of the
  36 recorded solutions, one per bundle atom.  Atom 1 spent `11.24 s`
  walking the rank from 6 to 26 because `_bundle_rank_hint` starts at 0;
  atom 2 started at 26.
- The rank hint is monotone
  (`self._bundle_rank_hint = max(self._bundle_rank_hint, maximum_bundle)`),
  so the cone never shrinks along the rollout, and the last atom's first
  call already costs `10.884 s`.
- Atom 0's accepted completion velocity is not among the 36: its message is
  `critical_schur:SQP3:Solved->SLSQP:0:...`, so the three recorded Clarabel
  SQP iterations did not converge and the velocity the rollout kept came
  from the SLSQP refinement that follows them inside `_solve_critical_schur`.
  Those three Clarabel calls cost `0.018 s` of the `145.819 s`.

## (b) Offline replay bench

New modules `paper_5/src/paper5/stability/cwrmf_solver_replay.py` (the
scoring and the candidates) and
`paper_5/src/paper5/stability/cwrmf_solver_replay_bench.py` (the driver).
Nothing in either runs inside a rollout.

Per instance the bench reports the replay wall time, the solver status, the
objective `0.5 x^T P x + q^T x`, the maximum cone violation of `b - A x`
over the declared cone blocks (entrywise on a zero or nonnegative block,
`||s_rest|| - s_0` on a second-order block, the negated minimum eigenvalue of
the unpacked matrix on a positive-semidefinite triangle block; nonpositive
means feasible), the minimum eigenvalue of the reconstructed 62-row Gram of
the candidate completion velocity, and the maximum absolute difference of
that velocity from the one the recorded solution implies.

The acceptance test is the bundle tier's own, applied to the full 62-row
Gram: with `G` the reconstructed Gram, `lambda_min = eigvalsh(G)[0]`, and

```
radial_residual     = max(0, |v|^2 - completion_radius_squared)
normalized_residual = max(0, safety - lambda_min, radial_residual,
                          r_prim, r_dual) / (1 + |lambda_min|)
accepted            = lambda_min >= safety - 1e-13
                      and normalized_residual <= 10 * solver_tolerance
```

evaluated with the strict profile's `(safety, solver_tolerance) =
(2e-12, 1e-10)` and with the balanced profile's `(1e-10, 1e-8)`.  The
recorded operands are the strict profile's, whose `psd_inflation` is `1e-11`
against the balanced profile's `1e-8`, so the balanced verdict is reported
twice: once on the recorded Gram, where the larger balanced safety margin is
compared against a Gram inflated by only `1e-11`, and once with
`1e-8 - 1e-11` added to the diagonal, which is the balanced profile's
`psd_inflation` substituted for the strict one.  The first form accepts
nothing at all: `balanced_accepted` and `recorded_balanced_accepted` are `0`
on all 36 instances for every one of the 22 candidates, Clarabel's own
recorded solutions included, so on strict-profile operands it separates
nothing and cannot be used as a gate.  The second is the one the tables use
and it is labelled `balanced (inflation substituted)`.

Every candidate is also scored against the verdict the rollout actually
took, because what drives the control flow is whether the same tier
iteration accepts.  `verdict agreement` counts instances where the
candidate's verdict equals the recorded solution's; the recorded solution
passes strict acceptance on 7 of the 36 (one per bundle atom) and the
substituted-inflation balanced test on 29 of the 36.

Validity gate: the `recorded` candidate -- the recorded settings, unchanged
-- reproduces all 36 recorded solutions bitwise
(`bitwise_equal_to_recorded = 36`, maximum velocity difference `0.000e+00`)
in `148.407 s`, against the `145.819 s` Clarabel reported for the same 36
calls online.  Every other candidate is measured against that replay
baseline, not against the online tier time.

SCS is called directly rather than through cvxpy.  The recorded instance is
already in SCS's standard form `min 0.5 x^T P x + q^T x` subject to
`A x + s = b`, `s` in the cone, so a cvxpy layer would only re-canonicalize
canonical operands, and cvxpy offers no way to warm start one problem from a
different problem's solution, which is what the task asks for.  The only
work the direct call needs is the row permutation from Clarabel's
upper-triangle column-major `sqrt(2)`-scaled packing to SCS's lower-triangle
column-major packing, plus SCS's fixed block order (zero, nonnegative,
second-order, semidefinite).  That permutation is checked two ways: a
repository test packs a random symmetric matrix and compares the permuted
vector entry by entry against the literal lower-triangle column-major
packing, and on recorded instance 3 SCS's returned solution has maximum cone
violation `5.303e-14` measured in Clarabel's own packing, against
`-7.910e-08` for Clarabel's solution on the same instance -- the violation
would be of order one if the permutation were wrong.  That instance also
shows the mechanism that decides the acceptance test: SCS reaches a slightly
lower objective (`2.262509e-15` against `3.487693e-14`) by sitting on the
cone boundary, while the guard needs the interior margin Clarabel's answer
carries.

### Bench table over the 36 recorded instances

Artifact:
`output/local_runs/paper_v_cwrmf_replay_bench_strict_t001_20260915_v1/bench.json`
(per-instance rows for every candidate).  One bench pass, 22 candidates,
36 instances each, ram_guard peak 2148 MB under a 4000 MB limit.  "strict
agree" and "balanced agree" count the instances where the candidate's
verdict equals the recorded solution's; "balanced" throughout is the
substituted-inflation form.

| candidate | total s | speedup | mean s | max s | strict accept (recorded 7) | strict agree | balanced accept (recorded 29) | balanced agree | bitwise | max velocity difference | max cone violation | min 62-row Gram eigenvalue |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `recorded` | 148.407 | 1.00x | 4.122 | 14.819 | 7 | 36/36 | 29 | 36/36 | 36/36 | 0.000e+00 | -1.867e-12 | -8.8055e-07 |
| `tolerance_1_decade` | 139.388 | 1.06x | 3.872 | 15.214 | 8 | 35/36 | 27 | 34/36 | 8/36 | 8.540e-04 | -1.508e-10 | -2.3612e-06 |
| `tolerance_2_decades` | 133.905 | 1.11x | 3.720 | 15.524 | 10 | 33/36 | 25 | 32/36 | 2/36 | 3.223e-03 | -2.665e-09 | -1.3941e-05 |
| `direct_solve_method_qdldl` | 760.620 | 0.20x | 21.128 | 79.448 | 5 | 32/36 | 27 | 34/36 | 1/36 | 2.719e-04 | -4.164e-12 | -8.8056e-07 |
| `direct_solve_method_faer` | 138.088 | 1.07x | 3.836 | 14.016 | 7 | 36/36 | 29 | 36/36 | 35/36 | 4.872e-10 | -1.867e-12 | -8.8055e-07 |
| `equilibrate_off` | 107.920 | 1.38x | 2.998 | 10.753 | 5 | 30/36 | 13 | 20/36 | 0/36 | 1.561e+00 | 3.549e-08 | -4.0789e-03 |
| `equilibrate_max_iter_2` | 136.894 | 1.08x | 3.803 | 11.869 | 9 | 34/36 | 29 | 36/36 | 0/36 | 2.997e-04 | -4.762e-10 | -4.4257e-07 |
| `presolve_off` | 134.830 | 1.10x | 3.745 | 13.525 | 7 | 36/36 | 29 | 36/36 | 36/36 | 0.000e+00 | -1.867e-12 | -8.8055e-07 |
| `static_regularization_off` | 103.770 | 1.43x | 2.883 | 10.443 | 8 | 33/36 | 25 | 32/36 | 0/36 | 1.005e+00 | 5.032e-08 | -2.5771e-03 |
| `static_regularization_1e_7` | 146.424 | 1.01x | 4.067 | 14.738 | 6 | 33/36 | 28 | 33/36 | 0/36 | 2.620e-04 | -3.098e-12 | -8.4316e-07 |
| `static_regularization_1e_10` | 108.640 | 1.37x | 3.018 | 11.071 | 8 | 33/36 | 26 | 33/36 | 0/36 | 4.903e-01 | -1.182e-09 | -1.2281e-03 |
| `chordal_decomposition_off` | 139.344 | 1.07x | 3.871 | 14.018 | 7 | 36/36 | 29 | 36/36 | 36/36 | 0.000e+00 | -1.867e-12 | -8.8055e-07 |
| `max_threads_2` | 104.964 | 1.41x | 2.916 | 10.429 | 7 | 34/36 | 29 | 36/36 | 6/36 | 2.242e-04 | -9.007e-12 | -8.8055e-07 |
| `max_threads_4` | 87.834 | 1.69x | 2.440 | 8.994 | 6 | 35/36 | 29 | 36/36 | 6/36 | 2.116e-04 | -6.439e-12 | -8.8055e-07 |
| `max_threads_8` | 84.623 | 1.75x | 2.351 | 8.430 | 6 | 35/36 | 29 | 36/36 | 6/36 | 2.255e-04 | -2.610e-11 | -8.8055e-07 |
| `scs_warm_1e_9` | 768.360 | 0.19x | 21.343 | 30.196 | 6 | 31/36 | 15 | 20/36 | 0/36 | 9.723e-04 | 3.814e-06 | -2.3253e-07 |
| `scs_warm_1e_9_polish_strict` | 868.262 | 0.17x | 24.118 | 44.026 | 9 | 34/36 | 29 | 36/36 | 32/36 | 1.355e-04 | -1.867e-12 | -8.8055e-07 |
| `scs_warm_1e_9_polish_balanced` | 962.648 | 0.15x | 26.740 | 44.013 | 7 | 36/36 | 29 | 36/36 | 36/36 | 0.000e+00 | -1.867e-12 | -8.8055e-07 |
| `reduced_bundle_1e_1` | 62.648 | 2.37x | 1.740 | 8.337 | 0 | 29/36 | 11 | 18/36 | 0/36 | 3.343e-04 | 6.112e-03 | -6.2186e-07 |
| `reduced_bundle_1e_2` | 22.536 | 6.59x | 0.626 | 3.994 | 0 | 29/36 | 3 | 10/36 | 0/36 | 3.386e-04 | 1.359e-02 | -1.3715e-06 |
| `reduced_bundle_1e_3` | 5.592 | 26.54x | 0.155 | 0.881 | 0 | 29/36 | 3 | 10/36 | 0/36 | 4.408e-04 | 9.473e-03 | -9.6982e-07 |
| `reduced_bundle_1e_6` | 3.187 | 46.57x | 0.089 | 0.204 | 0 | 29/36 | 3 | 10/36 | 0/36 | 5.591e-04 | 1.525e-02 | -1.5254e-06 |

The "min 62-row Gram eigenvalue" column is the minimum over all 36
instances, so it is dominated by the enlargement iterations that were
supposed to be infeasible; the acceptance columns are what separates the
candidates.

One row of that table is not an independent candidate: in this pass
`scs_warm_1e_9_polish_balanced` gated its polish on the unsubstituted
balanced test, which accepts nothing on strict-profile operands, so the
polish fired on all 36 instances and the row is Clarabel's own answer
every time.  The repaired gate is measured separately below.

### What the table says, by candidate group

**(2) Loosened tolerances are not the lever.** One decade buys `1.06x` and
two decades `1.11x`, both inside the machine's own spread, while the strict
verdict agreement falls to 35/36 and 33/36 and the balanced to 34/36 and
32/36.  The bundle tier already runs at `bundle_solver_tolerance = 1e-12`
and Clarabel takes 14 to 28 interior-point iterations on every bundle call
regardless; the cost is the factorization per iteration, which the tolerance
does not touch.  Note the direction of the acceptance counts: looser
tolerances raise the number of instances that *pass* (8 and 10 against the
recorded 7) while lowering agreement, because the enlargement iterations
stop short of the boundary and their Gram comes out less negative.

**(3) Of the algorithm settings, only one is free.** Forcing
`direct_solve_method='qdldl'` is `5.1x` slower (`760.620 s`), so Clarabel's
`auto` is already choosing `faer`, which the explicit `faer` run confirms
(`138.088 s`, 35 of 36 solutions bitwise, worst difference `4.872e-10`).
`chordal_decomposition_enable=False` changes nothing at all (36/36 bitwise):
the positive-semidefinite block is one dense cone with no sparsity to
exploit.  `presolve_enable=False` is the one free change: 36 of 36 solutions
bitwise identical at `134.830 s`, so the presolve pass finds nothing to
remove and its cost is pure overhead.  Everything else trades correctness
for speed and loses: `equilibrate_enable=False` is `1.38x` but moves a
returned velocity by `1.561` absolute and produces `InsufficientProgress`
and `NumericalError` statuses; `static_regularization_enable=False` is
`1.43x` at a `1.005` velocity difference and a `-2.5771e-03` Gram minimum.

**(4) Threads are the only real speedup, and they are not bitwise.**
`max_threads` 2, 4 and 8 give `1.41x`, `1.69x` and `1.75x` on this 14-core
M4 Pro (10 performance plus 4 efficiency cores).  Against the pinned
`max_threads = 1` recorded solutions: 6 of 36 are bitwise identical at every
thread count, and the worst velocity difference is `2.242e-04` at 2 threads,
`2.116e-04` at 4 and `2.255e-04` at 8.  The balanced verdict agrees 36/36 at
all three, and the balanced acceptance passes on exactly the recorded 29.
The strict verdict disagrees on one instance at 4 and 8 threads and on two
at 2 threads: at 4 and 8 threads the strict test passes on 6 of the 7
instances the recorded solution passes.

**(5) The warm-started first-order method is 5.2x slower, and it is
accepted on 15 of the 36.**  SCS at `eps_abs = eps_rel = 1e-9`, warm started
from the previous instance's returned `x` with `s = b - A x` and `y = 0`,
costs `768.360 s` against `148.407 s`, and its statuses include
`solved (inaccurate - reached max_iters)` and
`solved (inaccurate - reached time_limit_secs)` at the declared bounds
`max_iters = 100000`, `time_limit_secs = 30`.  Its answer passes the
substituted-inflation balanced acceptance test on 15 of the 36 instances
against the recorded solution's 29, agreeing with the verdict the rollout
took on 20 of the 36, and passes the strict test on 6 against the recorded 7
(31 of 36 agreement).  Those 15 sit at positive-semidefinite cone side
dimensions 12 to 78 across radial atoms 0 to 6, so they are not the small
cones alone, and on 14 of the 15 the recorded solution accepts as well.  The first submission said instead that the SCS answer
"satisfies the balanced acceptance test on none of the 36"; that sentence
was an artifact of the polish gate, not a measurement, and the repair is
below.  The mechanism is visible on instance 3: SCS reaches a lower
objective (`2.262509e-15` against Clarabel's `3.487693e-14`) by sitting on
the cone boundary at cone violation `5.303e-14`, where the guard needs the
interior margin Clarabel's `-7.910e-08` carries.  A hand-written
alternating-projection or ADMM on the same structure was not attempted: SCS
*is* an ADMM on exactly this splitting, it loses by a factor of 5 on time,
and under the repaired gate below it still needs a Clarabel solve on 19
of the 36 instances, so a slower Python implementation of the same
algorithm cannot change the verdict.

### The balanced polish gate, repaired and re-measured

The first pass's `scs_warm_1e_9_polish_balanced` called the acceptance test
with `profile='balanced'` and `inflation_shift` left at its default `0.0`, so
its polish gate was the unsubstituted balanced test -- the one that accepts 0
of 36 for every candidate measured, Clarabel's own recorded solutions
included.  That gate fired on all 36 instances by construction, so the row it
produced (`962.648 s`, 36 of 36 bitwise, 29 of 29 substituted-inflation
balanced accepts) measures Clarabel with an SCS warm-up in front of it and
separates nothing; the strict-gated candidate beside it is unaffected,
because the strict test is evaluated on the strict-inflation Gram the
operands carry and its polish fired on 32 of the 36.  The first submission
read that 36-of-36 polish as the sharp statement "the warm-started SCS
answer satisfies the balanced acceptance test on none of the 36"; the
bench's own `balanced_shifted_accepted = 15` for `scs_warm_1e_9`
contradicts it, and 15 is the measured number.

Repair: `WarmStartedScsCandidate.polish_inflation_shift`, default `0.0` --
which is what the strict-gated candidate keeps, since the strict test is
evaluated on the strict-profile operands the instances already carry -- and
the declared balanced-polish candidate now passes `BALANCED_INFLATION_SHIFT`,
so its gate is the same substituted-inflation balanced test every balanced
number in this report uses.  Re-measured over the same 36 instances with a
fresh same-pass `recorded` baseline.  Artifact:
`output/local_runs/paper_v_cwrmf_replay_bench_strict_t001_20260915_v3/bench.json`.

| candidate | total s | speedup | polish fired | strict accept (recorded 7) | strict agree | balanced accept (recorded 29) | balanced agree | bitwise |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `recorded` | 134.802 | 1.00x | -- | 7 | 36/36 | 29 | 36/36 | 36/36 |
| `scs_warm_1e_9_polish_balanced` | 720.502 | 0.19x | 19/36 | 7 | 32/36 | 30 | 35/36 | 19/36 |

Under the repaired gate the polish fires on 19 of the 36 instances, and
those 19 are exactly the rows that come back bitwise equal to the recorded
solution; on the other 17 the returned completion velocity differs from the
recorded one by at most `1.858e-04` and the maximum cone violation over the
36 rises from `-1.867e-12` to `4.973e-09`.  The substituted-inflation
balanced test accepts 30 of the 36, and those 30 contain all 29 the
recorded solution accepts, at 35 of 36 verdict agreement.  The strict test
also accepts 7, but not the recorded 7: only 5 instances are accepted by
both, so two the rollout accepted are rejected and two it rejected are
accepted, which is the 32 of 36 strict agreement.  `ram_guard` reported
peak 1782 MB under a 4000 MB limit for the pass.

The drop verdict is unchanged and is now stated on a gate that separates
something: the warm-started first-order method still needs a Clarabel solve
on 19 of the 36 instances to reach the balanced verdict this report uses,
and the combination costs `720.502 s` against the same pass's `134.802 s`
for Clarabel alone -- `5.3x`, against the `6.5x` the degenerate gate
produced, because 17 fewer Clarabel solves are paid for.

**(6) The reduced bundle is 2.4x to 46.6x faster and never acceptable.**
Dropping working-set directions whose eigenvalue in the recorded compressed
bundle base exceeds a declared fraction of that base's largest eigenvalue
shrinks the positive-semidefinite cone hard -- at `1e-3` the last atom's
cone falls from side dimension 80 to 44, and at `1e-6` from 80 to 12 -- and
the solve time falls with it, to `3.187 s` for all 36 instances at `1e-6`.
The strict acceptance test on the full 62-row Gram then passes on **zero**
of the 36, at every threshold tried, including the loosest (`1e-1`, which
still keeps 74 of 80 directions on the last atom and costs `62.648 s`).  The
maximum cone violation of the reduced answer measured against the *full*
recorded cone is `6.112e-03` at `1e-1` and `1.525e-02` at `1e-6`, against
`-1.867e-12` for the recorded answer: the dropped directions are exactly the
ones the full cone was constraining.  This is the measured refutation of
"the bundle is too large".

### Best candidate under each acceptance test

- Strict acceptance on all 36 instances, meaning 7 of 7 accepts and 36/36
  verdict agreement: `presolve_off`, `chordal_decomposition_off` and
  `direct_solve_method_faer` (the first pass's
  `scs_warm_1e_9_polish_balanced` also reaches 7 of 7, but only because its
  gate polished every instance with Clarabel).  The best
  by time in this pass is `presolve_off` (`134.830 s`, nominally `1.10x`),
  which is also bitwise on all 36; the second pass below shows that `1.10x`
  is the machine's drift and not the setting.
- Balanced acceptance on all 36 instances, meaning 29 of 29 accepts and
  36/36 verdict agreement: `max_threads_8` (`84.623 s`, `1.75x`) ahead of
  `max_threads_4` (`87.834 s`, `1.69x`) and `max_threads_2` (`104.964 s`,
  `1.41x`), then `equilibrate_max_iter_2`, `presolve_off`,
  `chordal_decomposition_off` and `direct_solve_method_faer`.

### The machine's own spread, measured

The `recorded` baseline of the pass above was measured while three other
heavy runs shared the machine; the later candidates ran with fewer.  A second
bench pass over the same 36 instances repeats `recorded` first and last, with
the shortlist in between.  Artifact:
`output/local_runs/paper_v_cwrmf_replay_bench_strict_t001_20260915_v2/bench.json`.

| candidate | total s | speedup against the pass's own `recorded` |
|---|---:|---:|
| `recorded` (first) | 135.669 | 1.00x |
| `presolve_off` | 135.711 | 1.00x |
| `presolve_off_faer` | 135.902 | 1.00x |
| `max_threads_8` | 82.623 | 1.64x |
| `max_threads_8_presolve_off` | 82.760 | 1.64x |
| `recorded_repeat` (last) | 135.746 | 1.00x |

The two `recorded` measurements bracket the pass at `135.669 s` and
`135.746 s`, a spread of `0.077 s` over thirteen minutes.  So the `1.10x` the
first pass credited to `presolve_off` was machine drift: `presolve_off` is
exactly free and exactly worthless for time, and disabling the presolve on
top of threads buys nothing either.  Only `max_threads` moves the number.

**The negative result, stated plainly: among every candidate the task named,
no conic solve that passes the strict full-Gram acceptance test on all 36
instances is measurably faster than the pinned single-threaded Clarabel call.
The only real speedup is multithreaded factorization, and it costs one of the
seven strict accepts while keeping all 29 balanced ones.**

## (c) Named solver profile

`CWRMFSettings.solver_setting_profile` (default `pinned_single_thread`, which
applies no override at all) and the driver flag `--solver-setting-profile`.
The named sets live in `CWRMF_SOLVER_SETTING_PROFILES` and are overlaid on
the tier's own Clarabel settings immediately before each of the four solves;
at the default the overlay returns without touching a field, which a
repository test asserts against every recorded settings field.

| profile | Clarabel setting set | qualified by |
|---|---|---|
| `pinned_single_thread` | none | the default; what the anchors were measured under |
| `presolve_off` | `presolve_enable = False` | best strict-passing candidate: 7 of 7 strict accepts, 36/36 verdict agreement, 36 of 36 solutions bitwise |
| `max_threads_8` | `max_threads = 8` | best balanced-passing candidate: 29 of 29 balanced accepts, 36/36 balanced verdict agreement, `1.64x` on the instances |

### Strict rollout with each, t = 0 to 0.01, h = 0.0025

Same configuration as the anchor throughout: strict numerical profile,
always-on guard, 4 SSPRK2 steps, 8 radial atoms, strong coupling
`lambda = 1.5`, `gamma = 0.5`, drive amplitude 1, phonon cutoff 16,
exact-correlated preparation, exact reference regenerated in process.

| quantity | anchor `..._final_strict_t001_h0025_20260914_v1` | `presolve_off` | `max_threads_8` |
|---|---:|---:|---:|
| autonomous wall seconds | `139.883` | `135.880` | `92.418` |
| Clarabel solve tier seconds / calls | `138.644` / 36 | `134.656` / 36 | `91.205` / 37 |
| all-coordinate scalar RMS against the exact cutoff-16 reference | `5.827261740618364e-05` | `5.827261740618364e-05` | `5.827261740618476e-05` |
| maximum absolute difference of the 384 carried coordinates from the anchor | -- | `0.000e+00` | `3.800246e-06` |
| maximum absolute difference of the 31 reported archive coordinates from the anchor | -- | `0.000e+00` | `2.088607e-15` |
| minimum carried unshifted 62-row Gram eigenvalue | `-3.8881605143449574e-12` | `-3.8881605143449574e-12` | `-4.242441681196736e-12` |
| minimum carried shifted lower bound | `5.105582279145534e-12` | `5.105582279145534e-12` | `4.747162291969484e-12` |
| maximum readable-rate residual | `1.4395728609345863e-07` | same | `2.9536214006129364e-06` |
| maximum completion correction norm | `2.953556872347944e-06` | same | `1.0201651066222305e-05` |
| maximum active critical modes / mean | 40 / 27.8 | 40 / 27.8 | 41 / 28.2 |
| driver-process peak resident set, `integration.peak_resident_bytes` (`RUSAGE_SELF`), MiB | `1609` | `1971` | `2443` |
| completed steps | 4 | 4 | 4 |

Artifacts:
`output/local_runs/paper_v_cwrmf_presolve_off_strict_t001_h0025_20260915_v1/`
and
`output/local_runs/paper_v_cwrmf_max_threads_8_strict_t001_h0025_20260915_v1/`.

`presolve_off` reproduces the anchor trajectory bitwise -- every array of
both npz files equal under `np.array_equal` -- so its `135.880 s` against the
anchor's `139.883 s` is the same load effect the bench pass measured, not a
speedup.  It is kept as the named strict-passing profile of record, and the
honest summary of it is that the strict acceptance test admits no faster
conic solve.

`max_threads_8` runs the strict rollout in `92.418 s` against the anchor's
`139.883 s` (`1.51x`) and against this session's bitwise-identical
default-path re-measurement of `135.880 s` (`1.47x`).  The all-coordinate
scalar RMS agrees with the anchor to `1.9e-14` relative; the 31 reported
archive coordinates agree to `2.088607e-15` absolute; the 384 carried
coordinates differ by at most `3.800246e-06`, which is the carried
completion block, not the reported observables.  The rollout takes 37
Clarabel calls instead of 36: the last atom needed one extra bundle
iteration and then a terminal polish that solved at iteration 0.  Its
maximum readable-rate residual rises from `1.44e-07` to `2.95e-06`, still
three decades under the declared `readable_rate_tolerance = 1e-5`, and its
minimum carried unshifted 62-row Gram eigenvalue is `-4.24e-12` against the
anchor's `-3.89e-12`, both inside the strict acceptance band
`schur_safety_margin - 1e-13 = 1.9e-12` after the `1e-11` inflation.  The driver process's own peak resident set
(`summary["integration"]["peak_resident_bytes"]`, from
`resource.getrusage(resource.RUSAGE_SELF)`, not a `ram_guard` process-tree
reading) rises from `1609 MiB` to `2443 MiB`.

Array by array against the anchor, `max_threads_8` (`np.array_equal`,
`maximum_atom_seconds` excluded):

| array | maximum absolute difference |
|---|---:|
| `times`, `exact_archive_coordinates`, `correction_iterations` | bitwise |
| `approximate_archive_coordinates` (5 x 31) | `2.088607e-15` |
| `carried_states` (5 x 384) | `3.800246e-06` |
| `minimum_unshifted_eigenvalues` | `2.761038e-12` |
| `minimum_shifted_lower_bounds` | `2.762656e-12` |
| `readable_rate_residuals` | `2.860611e-06` |
| `completion_correction_norms` | `7.248094e-06` |
| `velocity_margins` | `1.940012e-05` |
| `critical_modes` | 1 (one atom used 41 modes instead of 40) |

Not promoted: `max_threads_8` is off by default.  The record's own reason
stands -- the 2026-08-05 four-thread accelerator failed parity, and this one
is not bitwise either -- and plan step 7 gates raising `max_threads` on steps
5 and 6 showing the repair count rather than the repair cost dominating.
What the instances add is the number that gate was missing: on this machine
threads buy `1.51x` on the strict rollout while moving the reported archive
coordinates by `2.09e-15`.

## Files owned and changed

| file | change |
|---|---|
| `paper_5/src/paper5/stability/cwrmf_solver_instances.py` | new: `ConicInstanceRecorder`, `ConicInstance`, `load_instances`, `load_instance_context`, `build_solver_settings`, `build_cones`, `cone_specification` |
| `paper_5/src/paper5/stability/cwrmf_solver_replay.py` | new: scoring (`cone_blocks`, `maximum_cone_violation`, `objective_value`, `acceptance`, `score_candidate`, `summarize`) and the candidates (`ClarabelCandidate`, `ToleranceCandidate`, `ReducedBundleCandidate`, `WarmStartedScsCandidate`); repair round: `WarmStartedScsCandidate.polish_inflation_shift`, default `0.0`, passed into the polish gate's `acceptance` call |
| `paper_5/src/paper5/stability/cwrmf_solver_replay_bench.py` | new: the 25 declared candidates and the bench driver; repair round: `scs_warm_1e_9_polish_balanced` declares `polish_inflation_shift=BALANCED_INFLATION_SHIFT` |
| `paper_5/src/paper5/stability/apcm_carried_witness.py` | `CWRMF_SOLVER_SETTING_PROFILES`, `DEFAULT_SOLVER_SETTING_PROFILE`, `CWRMFSettings.solver_setting_profile`, `enable_instance_recording`, `instance_recorder`, `_apply_solver_setting_profile`, the recorder hook in `radial_atom`, and the record plus profile overlay at the four Clarabel call sites |
| `paper_5/src/paper5/stability/apcm_carried_witness_analysis.py` | `--record-conic-instances`, `--solver-setting-profile`, `summary["integration"]["recorded_conic_instances"]` |
| `paper_5/tests/cwrmf_solver_instances_test.py` | new: 5 tests, including the default-path bitwise test and the Gram reconstruction check |
| `paper_5/tests/cwrmf_solver_replay_test.py` | new: 13 tests, including the SCS permutation check, the profile-constant check, and the repair round's two gate tests (`test_polish_gate_uses_the_declared_inflation_shift`, `test_declared_balanced_polish_gate_carries_the_substituted_inflation`) |
| `paper_5/tests/test_apcm_carried_witness.py` | 30 -> 33 tests: the default solver setting profile touches no Clarabel field, the profile names are validated, and each named profile overlays only its own fields |

Suite: `cd paper_5 && PYTHONPATH=src python -m pytest -q -p no:cacheprovider`
gives 379 passed, 9 warnings in 130.01 s after the repair round (377
before it; the two added tests are the polish-gate pair).  The three files
above collect 33, 5 and 13 tests and run 51 passed in 3.03 s.

## Kept

- The conic instance recorder, off by default, `--record-conic-instances`;
  the strict rerun with it on reproduces the round-1 anchor trajectory
  bitwise.
- The offline replay bench and its 25 declared candidates (22 in the first pass, three added for the second).  Repair round: the balanced-polish candidate's gate is the substituted-inflation balanced test, declared through `polish_inflation_shift`, whose default `0.0` leaves the strict-gated candidate exactly as measured.
- `solver_setting_profile` on `CWRMFSettings` and the driver, default
  `pinned_single_thread` which applies no override, with
  `presolve_off` (strict-passing, bitwise on all 36) and `max_threads_8`
  (balanced-passing, `1.51x` on the strict rollout) as named options.

## Dropped

- Loosened Clarabel tolerances as a speedup: `1.06x` and `1.11x`, inside the
  machine spread, at 35/36 and 33/36 strict verdict agreement.
- `direct_solve_method='qdldl'`: `5.1x` slower than `auto`.
- `equilibrate_enable=False` and `static_regularization_enable=False`: `1.38x`
  and `1.43x`, but they move a returned completion velocity by `1.561` and
  `1.005` absolute and produce `InsufficientProgress` and `NumericalError`.
- `chordal_decomposition_enable=False`: exactly neutral, 36/36 bitwise, no
  time change; the positive-semidefinite block is one dense cone.
- The warm-started first-order method: SCS at `eps = 1e-9` is `5.2x` slower
  (`768.360 s` against `148.407 s` on the same 36 instances) and its answer
  passes the substituted-inflation balanced acceptance test on 15 of the 36
  against the recorded solution's 29.  With the repaired balanced polish
  gate the Clarabel solve fires on 19 of the 36 and the combination costs
  `720.502 s` against `134.802 s` in the same pass, `5.3x`.  A hand-written
  alternating-projection or ADMM was not attempted, because SCS is an ADMM
  on this splitting and a Python implementation of the same algorithm
  cannot reverse a 5x loss.
- The reduced bundle at every threshold tried: zero strict accepts of 36,
  and a maximum cone violation against the full recorded cone of `6.112e-03`
  even at the loosest threshold, which keeps 74 of the last atom's 80
  directions.

## Blockers and what this leaves for the next round

- The strict acceptance test admits no faster conic solve among the declared
  candidates.  Making the strict guard fast therefore needs the *number* of
  solves to fall, not the cost of one.  The instances point at where: 26 of
  the 33 bundle calls are enlargement iterations whose solution is
  discarded, and `_bundle_rank_hint` starts at 0, so atom 1 spent `11.24 s`
  walking the cone side dimension from 12 to 52 before the atom that used it
  started at 52.  Seeding the hint from the endpoint Gram spectrum, or
  carrying it across a restart, is the untried lever and it is arithmetic,
  not a scientific decision.
- `max_threads_8` is qualified only against the balanced acceptance test and
  only on this 14-core M4 Pro.  Promoting it needs D3 answered (whether the
  balanced profile is the reported cone) and a rerun on whatever machine the
  reported numbers come from; plan step 7 also gates it on steps 5 and 6.
- Whether the `1.51x` survives to `t=0` to `0.5` is unmeasured here: the
  recorded instances stop at `t=0.01`, where the bundle side dimension has
  reached 80 of a possible 124, and the record's `t=0.5` strict run needed
  up to 57 of 61 critical modes.
- Every balanced number in this report is evaluated on strict-profile
  operands with the balanced profile's `psd_inflation` substituted for the
  strict one (`1e-8 - 1e-11` added to the reconstructed Gram's diagonal).
  Without that substitution the balanced acceptance test accepts nothing on
  these operands, Clarabel's own recorded solutions included, which is the
  defect the repair round removed from the polish gate.  A balanced verdict
  that does not rest on a stated substitution needs its own recorded
  rollout in the balanced profile; that run has not been made.

## Repair round: what changed against the reviewed submission

| review issue | disposition | evidence |
|---|---|---|
| material: the SCS balanced-acceptance claim contradicts `bench.json` | both remedies taken -- the prose now reports the measured 15 of 36 accepts and 20 of 36 verdict agreement under the substituted-inflation balanced test, and the polish gate is repaired in code (`polish_inflation_shift`, default `0.0`) and the balanced-polish candidate re-measured | `output/local_runs/paper_v_cwrmf_replay_bench_strict_t001_20260915_v1/bench.json` for the 15, `..._v3/bench.json` for the repaired gate (`720.502 s`, polish on 19 of 36, 30 balanced accepts containing the recorded 29, 7 strict accepts of which 5 coincide) |
| factual: the bundle iteration range is 14 to 28 | corrected in both places | the 33 `bundle_cone` rows of `..._record_strict_t001_h0025_20260915_v1/conic_instances/instances.json`: 14 once, then 17 to 28; the three `critical_schur` rows are 12 each |
| factual: the recorder write is outside the Clarabel tier | corrected; the extra `8.163 s` is attributed to contention inside the solver | `clock.add("clarabel_solve", ...)` precedes `self._instance_recorder.record(...)` at all four call sites; `unaccounted_seconds` `0.2465` to `0.3983`; the 36 recorded `solve_time` values sum to `145.819 s` against the anchor's `138.644 s` tier |
| cosmetic: rows labelled `ram_guard peak` | relabelled as the driver process's own `integration.peak_resident_bytes` (`RUSAGE_SELF`), in MiB | `apcm_carried_witness_analysis.py` builds that field from `resource.getrusage(resource.RUSAGE_SELF).ru_maxrss` |

No default, acceptance constant, numerical profile or scientific rule was
touched by the repair.  `test_polish_gate_uses_the_declared_inflation_shift`
pins the new field's effect in both directions on a synthetic instance whose
reconstructed Gram has minimum eigenvalue `0.0`, and
`test_declared_balanced_polish_gate_carries_the_substituted_inflation` pins
the declared candidates: balanced carries `BALANCED_INFLATION_SHIFT`, strict
carries `0.0`, and the dataclass default is `0.0`.  Full `paper_5` suite 379
passed after the repair, 377 before.
