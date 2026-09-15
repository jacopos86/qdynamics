# Review: cone-correction-speedup (plan step 8), 2026-09-14

Reviewer: independent. No source file was edited by this review. Verification
artifacts are under `output/local_runs/paper_v_cone_review_20260914/`
(git-ignored, inside the section 7 allowlist).

## Verdict

Pass. Every gate the implementer claimed was reproduced independently, and the
default path is byte-identical to the pre-change module across all five public
correction routines. Two code findings and three reporting findings are listed
below; none of them moves a recorded number, because all new behaviour is off
by default.

## Scope

`git status --short` and `git diff --stat` in the worktree show eight modified
files. Two are the implementer's: `paper_5/src/paper5/stability/cone_correction.py`
(+692/-134 net over 906 changed lines in the two files) and
`paper_5/tests/test_cone_correction.py`. The other six
(`apcm_carried_witness.py`, `apcm_carried_witness_analysis.py`,
`apcm_positive_extension.py`, `apcm_projective_guard.py`, `projected_apcm.py`,
`test_apcm_carried_witness.py`) and the two untracked
`apcm_projective_event_guard_analysis.py` / `test_apcm_projective_event_guard.py`
belong to the sibling CWRMF and projective-guard agents; grepping their diffs
for `cone_correction`, `halfspace`, `SOLVER_COUNTERS`, `cut_harvest` and
`memoize_corrections` returns nothing, so the cone work is contained to the two
owned files. The written artifacts
(`agent_guidance/paper-v/overnight/cone-correction-speedup-20260914.md`,
`output/local_runs/paper_v_cone_ref_t2_20260914_v1/**`) are inside the section 7
allowlist. No scope violation.

## Gates rerun

| gate | command | result |
|---|---|---|
| owned test file | `cd paper_5 && PYTHONPATH=src ... pytest -q -p no:cacheprovider tests/test_cone_correction.py` | 29 passed in 6.66 s |
| whole `paper_5` suite | same without a path argument | 326 passed, 9 warnings, 100.17 s |

## Parity, re-established from the pre-change module

The stored reference was not taken on trust. `git show HEAD:paper_5/src/paper5/stability/cone_correction.py`
(2049 lines, the length the task names) was written into a copy of the `paper5`
package at `output/local_runs/paper_v_cone_review_20260914/prechange_src/`, and
both pinned runs were executed twice, once against that tree and once against
the worktree tree, with the implementer's harness
`output/local_runs/paper_v_cone_ref_t2_20260914_v1/run_pinned_energy_neutral_barrier.py`.

Energy-neutral matrix barrier on the ten (dN, dA) control coordinates
(`structured_boson_barrier_correction`, `energy_neutral=True`, through
`closed_cone_projected_rhs`) at lambda_ep = 1.5, gamma = 0.5, drive amplitude 1,
exact ground-state contraction at phonon cutoff 16, activation_margin 1e-5,
target_flux 0, barrier_rate 5, DOP853 rtol 1e-9 atol 1e-11 maximum step 0.05,
t = 0 to 2 sampled every 0.05: `states`, `times`, `energies`,
`boson_moment_minima` and `electron_minima` are `np.array_equal` True between my
pre-change and post-change runs, my pre-change run and the implementer's stored
reference, and my post-change run and the implementer's post-change run. 749
right-hand-side evaluations and 484 half-space projections in every case.
Minimum bosonic moment eigenvalue 2.893142825017192e-05 over t = 0 to 2, which
is the working notes' pinned value 2.89314e-5 (notes:1859, whose horizon is
t = 0 to 20).

Energy-neutral joint-Gram barrier on the 31 closed-state coordinates
(`structured_electron_phonon_barrier_correction` through
`closed_electron_phonon_cone_projected_rhs`, `preserve_correlation_trace` true,
cone_tolerance 1e-8, projection_tolerance 1e-12, euclidean correction metric),
same physical and integrator settings, t = 0 to 0.2 sampled every 0.01: the same
five arrays are `np.array_equal` True in all three comparisons, at 16,700
right-hand-side evaluations and 318,122 half-space projections in both runs.

Byte-level probe on the recorded many-mode boundary state (the 31 coordinates in
`test_cone_correction.py`, derivative `closed_scalar_rhs(28.545, ...)`) plus the
cutoff-12 exact ground contraction at t = 0.3, comparing the hex image of
`correction_coordinates` pre against post: `structured_joint_barrier_correction`
(cutting_plane and direct_eigenvalue, and its `constraint_count`),
`structured_full_state_joint_barrier_correction`,
`structured_electron_phonon_barrier_correction` (euclidean, frobenius,
direct_eigenvalue, and their `constraint_count`), `structured_boson_barrier_correction`,
`structured_boson_cone_correction` and `structured_boson_velocity_lift` are all
identical. This covers the default paths the two pinned runs do not exercise,
in particular the restructured single-mode cut loop and the `np.block`
substitution.

## Defaults

`DEFAULT_HALFSPACE_BACKEND = "slsqp"` (`cone_correction.py:41`), `cut_harvest`
defaults to `"single_mode"` (`:970`), `memoize_corrections` defaults to `False`
on all four right-hand-side factories. `_METRIC_COORDINATE_MAPS`
(`:410-429`) evaluates the Cholesky and the triangular solve of two fixed
matrices at import instead of per call, same operands; the joint-Gram basis
hoist (`:1612-1618`) replaces
`structured_electron_phonon_moment_velocity_lift(state_array, e_i)` by
`electron_phonon_moment_derivative(matrix_state, _CLOSED_STATE_VELOCITY_BASIS[i])`,
which is that function's body (`:473-479`) applied to the object
`matrix_state` already built at `:1526` from the same `state_array`.

## Wall time

Measured on this machine while the sibling overnight agents were running, so
these are load-dependent and should not be quoted as a fixed percentage.

| run | pre-change | post-change |
|---|---:|---:|
| joint-Gram barrier, t = 0 to 0.2 (mine) | 127.56 s | 106.43 s (16.6 percent less) |
| joint-Gram barrier, t = 0 to 0.2 (implementer's) | 145.62 s | 110.74 s (24.0 percent less) |
| boson matrix barrier, t = 0 to 2 (mine) | 0.2150 s | 0.2157 s |

The direction is confirmed and the trajectory is unchanged; the size of the
reduction differs by machine load, and the ten-coordinate boson run is too short
for its 0.2426 s to 0.2261 s figure to separate from noise.

## Findings

1. `_active_set_minimum_norm_point` declares success on feasibility alone.
   `cone_correction.py:663-696`: when the `while iterations < limit` loop
   exhausts its limit rather than breaking on the optimality test, the last
   computed `point` is returned with `converged=True` as soon as
   `maximum_violation <= 10.0 * tolerance`, with no check that the working-set
   multipliers were nonnegative. A feasible but non-minimum-norm point can
   therefore be returned as the declared correction. The same exit is reached
   when the final iteration popped a constraint and `point` is stale from the
   previous working set. This is the likely source of the 1.1800812133903852e-05
   maximum relative distance between the two backends' minimum-norm points
   recorded over 16,646 corrections in
   `output/local_runs/paper_v_cone_ref_t2_20260914_v1/halfspace_backend_replay.json`,
   and it is what the docstring's uniqueness argument (`:619-629`) does not
   cover: uniqueness of the minimizer is not a certificate that this walk found
   it. The default is unaffected, so no recorded number moves; fix before
   `DEFAULT_HALFSPACE_BACKEND` is ever flipped.
2. `cone_correction.py:693-694` stores `warm_state.active = tuple(working)`
   even on the path that falls back to SLSQP, so the next pass warm-starts from
   a working set that did not certify feasibility.
3. The implementer's report says "16,646 solved by the single-cut closed form,
   300,631 by the warm-started active-set walk, 29,336 falling back to SLSQP".
   The 29,336 fallbacks are a subset of the 300,631 walks, not a third disjoint
   class. The avoided-call arithmetic is right: 317,277 - 29,336 = 287,941.
4. "The stated 1e-12 gate cannot be met by any change of minimum-norm-point
   solver" is stronger than what was measured, which is that SLSQP terminates on
   ftol 1e-12 and that this active-set backend moves the integrated joint-Gram
   trajectory by 6.286099962725197e-09 relative to the largest coordinate over
   t = 0 to 0.2.
5. The task asked for an existing driver in `pipelines/open_dynamics`. The
   implementer wrote a harness instead. This is justified and the deviation is
   recorded in the harness docstring:
   `pipelines/open_dynamics/run_high_coupling_controller_comparison.py` is a
   fixed-step driver (`--time-step`) with a `GaussianSineDrive` protocol and no
   DOP853, rtol, atol or maximum-step controls, so it cannot run the pinned
   protocol. I checked the harness against the producer of the notes' pinned
   table, `paper5.stability.report._cone_correction_audit` (`report.py:458-500`)
   with `strong_parameters` (`report.py:1936-1940`) and `closed_exact_state`
   (`report.py:2038-2041`): parameters, initial state, right-hand side,
   integrator, tolerances and maximum step all match.

## Gate reading

The task's gate names "the energy-neutral matrix barrier", which is the
ten-coordinate boson controller of the working notes (notes:1794-1873, the
correction confined to N and A under dN_00 + dN_11 = 0). That run was executed
over the full stated horizon t = 0 to 2 and is byte-identical, so the gate is
met as written. The 31-coordinate joint-Gram controller was run to t = 0.2 only,
which the implementer states and attributes to the fifteen-minute per-run
ceiling; at 16,700 right-hand-side evaluations for t = 0.2 that attribution is
consistent with the measured cost.
