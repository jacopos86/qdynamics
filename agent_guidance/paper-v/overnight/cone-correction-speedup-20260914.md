# cone-correction-speedup (2026-09-14)

Plan step 8, `agent_guidance/paper-v/paper-v-state-and-plan-20260914.md:268-283`.
Branch `claude/paper-v-routes-20260914`, worktree
`/Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914`.
Not committed; the parent session commits.

## Negative results first

1. **Memoizing the correction on the exact `(state, derivative)` pair never
   hits.** DOP853 at rtol 1e-9, atol 1e-11, maximum step 0.05 evaluated the
   right-hand side 749 times over t = 0 to 2 for the energy-neutral matrix
   barrier and 218 times over t = 0 to 0.05 for the joint-Gram barrier, and in
   both runs every evaluation was at a distinct `(state, derivative)` pair:
   749 misses and 0 hits, 218 misses and 0 hits. DOP853 reuses the previous
   step's final stage rather than recomputing it, and a rejected step retries
   at different stage points, so there is nothing at an identical state to
   reuse. The memo is implemented and tested but is off by default.

2. **The 1e-12 gate cannot be met by any change of minimum-norm-point solver.**
   Running the joint-Gram barrier over t = 0 to 0.2 with the active-set backend
   moves the trajectory 7.698e-9 in absolute coordinate value, 6.286e-9
   relative to the largest coordinate, and changes the evaluation count from
   16,700 to 17,336. SLSQP terminates on `ftol = 1e-12` in the objective, which
   leaves its returned point further than 1e-12 from the exact minimum-norm
   point; the active-set walk lands closer. The gate is therefore satisfied
   only by the changes that leave the floating-point path alone, and those are
   what is on the default path.

3. **The dominant remaining cost is outside the files this task owns.** After
   the change, `electron_phonon_moment_derivative`
   (`paper_5/src/paper5/stability/matrix_reference.py:1162`) accounts for 0.622 s
   of the 1.329 s profiled cost of the joint-Gram route over t = 0 to 0.05: 32
   calls per right-hand-side evaluation, 142,552 `ndarray.trace` calls and
   13,596 `np.block` calls over 206 evaluations.

## What changed

All in `paper_5/src/paper5/stability/cone_correction.py` (2049 -> 2482 lines)
and `paper_5/tests/test_cone_correction.py` (880 -> 1202 lines). Line numbers
are post-change.

### Bitwise-neutral, on the default path

| change | file:line | what it removes |
|---|---|---|
| the 31-fold joint-Gram basis is built by applying `electron_phonon_moment_derivative` to the existing `matrix_state` and to the module constant `_CLOSED_STATE_VELOCITY_BASIS` | `cone_correction.py:1612-1621` | 31 rebuilds of the matrix state and 31 rebuilds of the lifted unit-coordinate matrix states per correction |
| the uncorrected joint-Gram velocity uses the same `matrix_state` | `cone_correction.py:1538-1541` | one more matrix-state rebuild per correction |
| the identity-metric Cholesky factor and its triangular solve against the identity become module constants for both declared metrics | `cone_correction.py:410-429`, used at `1597` | one 31x31 Cholesky and one 31x31 triangular solve per correction |
| the corrected barrier matrices are cached on the exact reduced point, shared by the constraint function, the Jacobian and the loop's exit test | `cone_correction.py:862-882`, `1624-1642` | one 31x7x7 tensordot and two matrix lifts per repeated point |
| `structured_boson_velocity_lift` assembles the 4-by-4 block by slice copy instead of `np.block` | `cone_correction.py:295-301` | the recursive shape check, whose Python cost exceeded the four copies |

### Named options, defaults untouched

| option | where | default |
|---|---|---|
| `halfspace_backend` on `_project_origin_onto_halfspaces`, `structured_boson_barrier_correction`, `structured_joint_barrier_correction`, `structured_electron_phonon_barrier_correction` and the matching right-hand sides | `cone_correction.py:712-752` | `None`, resolving to `DEFAULT_HALFSPACE_BACKEND = "slsqp"` (`:41`) |
| `"active_set"`: warm-started dual active-set minimum-norm-point walk, with the closed form `y = max(0, b) n / (n . n)` when one half-space has accumulated, and SLSQP as the declared fallback when the walk does not certify feasibility | `cone_correction.py:611-709`; carried working set in `HalfspaceWarmState` (`:566-574`), created per constraint-generation loop at `:1153`, `:1734`, `:1989` | not selected by any caller |
| `cut_harvest="all_violated"` on `structured_joint_barrier_correction`: every eigenmode below the cone tolerance in every sector enters on the same pass, matching what the electron--phonon routine already does | `cone_correction.py:1208-1247` | `"single_mode"` |
| `memoize_corrections` on all four right-hand-side factories | `cone_correction.py:2210`, `2289`, `2362`, `2438`; cache at `:82-129` | `False` |
| `SOLVER_COUNTERS` (`cone_correction.py:45-76`): solver call census, reset per accounting window | always counting | changes no number |

The minimum-norm point of a nonempty polytope is unique, because the objective
`1/2 ||y||^2` is strictly convex and the feasible set is convex. That is why
the active-set backend solves the same problem the recorded SLSQP backend
solves and why the cross-backend norm-agreement test is the right acceptance
test for it.

## Gate results

### Test suite

`cd paper_5 && PYTHONPATH=src python -m pytest -q -p no:cacheprovider`:
**326 passed** in 100.90 s, including `tests/test_cone_correction.py` at
**29 passed** (22 before; seven added).

Added tests: `structured_boson_velocity_lift` byte-equal to the `np.block`
assembly; cross-backend minimum-norm agreement at rtol 1e-8 over 120 random
polytopes (86 converged on both backends); the closed-form single-cut solution
exactly; warm-started agreement as the cut set grows one half-space at a time
through 14 cuts; the declared correction byte-identical when `"slsqp"` is named
explicitly and equal in norm at rtol 1e-8 for `"active_set"`, on the pinned
strong-coupling state; `cut_harvest="all_violated"` byte-identical for the
default and equal in norm at rtol 1e-6 on the many-mode boundary state; the
memoized right-hand side byte-identical to the unmemoized one with one recorded
hit and one recorded miss.

### Pinned run 1: energy-neutral matrix barrier, ten (dN, dA) controls

`structured_boson_barrier_correction` with `energy_neutral=True` through
`closed_cone_projected_rhs`, at `lambda_ep = 1.5`, `gamma = 0.5`, drive
amplitude 1, exact ground-state contraction at phonon cutoff 16,
`activation_margin = 1e-5`, `target_flux = 0`, `barrier_rate = 5`, DOP853
rtol 1e-9, atol 1e-11, maximum step 0.05, over t = 0 to 2, sampled every 0.05.

| quantity | before | after |
|---|---:|---:|
| wall seconds | 0.2426 | 0.2261 |
| right-hand-side evaluations | 749 | 749 |
| half-space projections | 484 | 484 |
| minimum bosonic moment eigenvalue over t = 0 to 2 | 2.893142825017192e-05 | 2.893142825017192e-05 |

Sampled states, times, eigenvalue series and energies are **byte-identical**
(maximum absolute coordinate difference 0.0). The minimum bosonic moment
eigenvalue reproduces the working notes' pinned t = 20 value 2.89314e-5
(`MATH/paper_facing/paper_V_high_u_gkba/stability_quantum_workflow_working_notes.md:1859`).

### Pinned run 2: energy-neutral joint-Gram barrier, 31 closed-state controls

`structured_electron_phonon_barrier_correction` through
`closed_electron_phonon_cone_projected_rhs`, same physical settings, plus
`preserve_correlation_trace=True`, `cone_tolerance = 1e-8`,
`projection_tolerance = 1e-12`, euclidean correction metric,
`subtract_initial_residual=False`, over t = 0 to 0.2, sampled every 0.01.

| quantity | before | after |
|---|---:|---:|
| wall seconds | 145.62 | 110.74 |
| right-hand-side evaluations | 16,700 | 16,700 |
| half-space projections | 318,122 | 318,122 |
| minimum bosonic moment eigenvalue over t = 0 to 0.2 | 2.0646790110955617e-05 | 2.0646790110955617e-05 |

Sampled states, times, eigenvalue series and energies are **byte-identical**.
Wall time is 24.0 percent lower. Under `cProfile` over t = 0 to 0.05 the same
route falls from 1.986 s to 1.329 s.

The horizon is t = 0 to 0.2 and not t = 0 to 2 because this controller does not
finish t = 0 to 2 inside the fifteen-minute ceiling: the attempt was stopped at
600 s. The step size collapses under the barrier, so t = 0 to 0.2 already costs
16,700 evaluations, 19.05 half-space projections per evaluation.

### Half-space backend replay at identical operands

The joint-Gram route was integrated once with the recorded SLSQP backend over
t = 0 to 0.2, recording all 16,646 `(state, derivative)` pairs the right-hand
side was asked to repair; each backend was then timed over that same list, so
the comparison is of solver cost and not of two trajectories.

| quantity | slsqp | active_set |
|---|---:|---:|
| wall seconds for 16,646 corrections | 109.91 | 92.55 |
| half-space projections | 317,277 | 317,277 |
| of which solved by the single-cut closed form | 0 | 16,646 |
| of which solved by the warm-started walk | 0 | 300,631 |
| of which fell back to SLSQP | 0 | 29,336 |
| corrections that failed the convergence contract | 0 | 0 |

**SLSQP calls avoided: 287,941 of 317,277, 90.8 percent.** Wall time falls
15.8 percent at the same operands.

Agreement between the two backends over those 16,646 corrections: maximum
relative difference in the correction norm 3.082250326674586e-10; maximum
relative distance between the two minimum-norm points 1.1800812133903852e-05;
maximum difference in any single correction coordinate 9.21346790097588e-07.
On 400 random polytopes of 1 to 11 half-spaces in 2 to 13 coordinates the worst
relative distance between the backends is 5.197399383858669e-13.

## Kept

- Every bitwise-neutral hoist and cache listed above. They carry the whole
  24.0 percent and cost no accuracy.
- The `active_set` backend as a named, non-default backend with SLSQP as its
  declared fallback, and the closed-form single-cut path inside it.
- `cut_harvest="all_violated"` as a named option on the joint routine. On the
  many-mode boundary state it accumulates 114 half-spaces against 98 for the
  default and reaches the same corrected correction norm to 1e-6 relative.
- `memoize_corrections` as a named option, with its measured zero hit rate
  recorded rather than the option removed: a caller that evaluates the same
  correction twice (an offline audit that re-scores sampled times, a
  fixed-step scheme that reuses a stage) still gets the reuse.
- `SOLVER_COUNTERS`, which is how every call count in this report was measured.

## Dropped

- **Making `active_set` the default.** It fails the stated 1e-12 gate (6.286e-9
  relative) and, integrated rather than replayed, buys only 113.50 s against
  110.74 s because the different trajectory costs 636 more evaluations.
  Flipping `DEFAULT_HALFSPACE_BACKEND` (`cone_correction.py:41`) is a one-line
  decision if the parent decides the 1e-12 gate should be relaxed to the
  measured 6e-9.
- **Two attempts to remove the 9.3 percent active-set fallback rate.**
  (i) Tightening the entering tolerance to the constraint scale, widening the
  iteration limit and dropping a dependent row when the working-set face solve
  missed its own face: fallbacks rose to 33,209 and the integrated run cost
  126.91 s against 113.50 s. (ii) Solving the face equations by least squares
  when the Gram route missed the face: the fallback count was unchanged at
  29,336 and the replay cost rose from 92.55 s to 96.10 s. Both reverted. The
  fallback is not caused by a dependent working set.
- **Fusing the eigenvalue and Jacobian eigendecompositions in the
  `direct_eigenvalue` backends.** `np.linalg.eigvalsh` and
  `np.linalg.eigh(...)[0]` call LAPACK with `jobz = 'N'` and `jobz = 'V'` and
  are not guaranteed to agree in the last bits, so sharing one decomposition
  between the constraint function and the Jacobian would move the default path.
  What is shared instead is the matrix assembly, which is a copy and is exact.
  The `direct_eigenvalue` backends are also off the pinned path: both pinned
  runs use `cutting_plane`.
- **Vectorizing the 31-fold mode-response contraction**
  `[real(vdot(mode, basis @ mode)) for basis in joint_velocity_basis]`. A
  batched `matmul` or `einsum` changes the summation order and therefore the
  last bits.

## Blockers

1. The next factor is in `paper_5/src/paper5/stability/matrix_reference.py`
   (`electron_phonon_moment_derivative:1162`,
   `_electron_fluctuation_moment_derivative:1084`,
   `_correlation_pauli_block:1119`), which this task does not own. It is 47
   percent of the profiled joint-Gram cost after this change, spent in 142,552
   `ndarray.trace` calls and 13,596 `np.block` calls per 206 right-hand-side
   evaluations. The same slice-copy substitution that worked here applies, and
   the 31-fold application of a linear map to a fixed set of basis matrices can
   be assembled once per state.
2. Plan step 8's own gate asks for a pinned t = 20 run; the joint-Gram
   controller cannot reach t = 2 inside fifteen minutes on this machine, let
   alone t = 20. Step 9's declared t = 100 energy-neutral calculation is
   feasible only for the ten-coordinate matrix barrier (0.226 s to t = 2), not
   for the 31-coordinate joint-Gram barrier.
3. `detect_changes` was not run: this worktree is not in the GitNexus index
   (the index covers the main checkout only), so a zero from it would mean
   unseen, not clean. Impact analysis for the edited symbols was run against
   the main checkout's index and is recorded below.

## Impact analysis (read-only, main checkout index)

| symbol | risk | impacted | direct callers |
|---|---|---:|---|
| `_project_origin_onto_halfspaces` | LOW | 22 | `structured_boson_barrier_correction`, `structured_joint_barrier_correction`, `structured_electron_phonon_barrier_correction` |
| `structured_electron_phonon_barrier_correction` | MEDIUM | 23 | `closed_electron_phonon_cone_projected_rhs.rhs`, `electron_phonon_analysis._exact_defect_metrics`, `electron_phonon_analysis.integrate_closed_rk4`, and four more |
| `structured_boson_velocity_lift` | MEDIUM | 57 | six direct |
| `structured_joint_barrier_correction` | LOW | 1 | `closed_joint_cone_projected_rhs.rhs` |

The MEDIUM verdicts are the reason both pinned runs were required to be
byte-identical rather than merely close, and the full 326-test suite was run.

## Artifacts

All under
`/Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_cone_ref_t2_20260914_v1/`:

| path | contents |
|---|---|
| `run_pinned_energy_neutral_barrier.py` | the pinned-run harness; settings copied from `paper5.stability.report._cone_correction_audit` with `correction_mode="energy_neutral"`, which cannot itself run at final time 2 because its post-pulse window starts at t = 4 |
| `replay_halfspace_backends.py` | the identical-operand backend replay |
| `pinned_energy_neutral_barrier.npz`, `summary.json` | pre-change ten-coordinate reference, t = 0 to 2 |
| `gram_t02/` | pre-change 31-coordinate reference, t = 0 to 0.2 |
| `final_boson_default/`, `final_gram_default/` | post-change default path, both byte-identical to the references above |
| `gram_t02_after_active_set/` | 31-coordinate run with `halfspace_backend="active_set"` |
| `boson_memoized/`, `gram_memoized/` | memoization hit-rate census |
| `halfspace_backend_replay.json` | the replay table above |

Every process ran under
`pipelines/shell/ram_guard.py --limit-mb 4000`; the largest recorded peak was
119 MB, one process at a time.
