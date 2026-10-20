# CWRMF cone-guard speedup (key `cwrmf-cone-speedup`), 2026-09-14

Worktree `/Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914`,
branch `claude/paper-v-routes-20260914`.  Plan steps 2, 3, 4 of
`agent_guidance/paper-v/paper-v-state-and-plan-20260914.md`.  Nothing committed;
strict and balanced defaults unchanged.

## Negative result first

The strict always-on guard's wall time is the Clarabel solve, not the Python
assembly the plan's code reading ranked first.  Over t = 0 to 0.01 at
h = 0.0025 (4 SSPRK2 steps, 8 radial atoms, strong coupling, cutoff-16
correlated preparation, strict profile) the rollout took 145.17 s and split:

| tier | seconds | share of rollout | calls |
|---|---:|---:|---:|
| Clarabel solve | 142.350 | 98.1% | 36 |
| coefficient assembly and svec packing | 2.345 | 1.6% | 36 |
| 62-row eigendecomposition | 0.049 | 0.0% | 62 |
| unguarded velocity evaluation | 0.149 | 0.1% | 8 |
| completion retraction | 0.000 | 0.0% | 0 |
| unaccounted | 0.280 | 0.2% | |

Artifact: `output/local_runs/paper_v_cwrmf_instr_strict_t001_h0025_20260914_v1/summary.json`.

The matched no-guard rollout over t = 0 to 0.05 at h = 0.0025 (20 steps, 40
atoms) took 0.928 s and split the other way: unguarded velocity evaluation
0.7386 s (79.6%, 0.01847 s per atom), 62-row eigendecomposition 0.0833 s
(9.0%), unaccounted 0.1065 s.  Artifact:
`output/local_runs/paper_v_cwrmf_instr_noguard_t005_h0025_20260914_v1/summary.json`.

Reading: the three bitwise-neutral accelerations named in plan step 3 that
touch assembly and velocity evaluation address at most 1.6% of the strict
guarded atom, and about 89% of the unguarded atom, which is the
event-triggered guard's steady state.

## Reference runs (pre-change, saved first)

| run | directory | autonomous wall seconds | peak child RSS |
|---|---|---:|---|
| no guard, t = 0 to 0.05, h = 0.0025 | `output/local_runs/paper_v_cwrmf_ref_noguard_t005_h0025_20260914_v1` | 1.2864 | 682 MB (process max RSS) |
| strict, t = 0 to 0.01, h = 0.0025 | `output/local_runs/paper_v_cwrmf_ref_strict_t001_h0025_20260914_v1` | 143.511 | 1980 MB (ram_guard peak) |

Parity harness: every array in `autonomous_trajectory.npz` and
`trajectory.npz` except `maximum_atom_seconds` compared with
`np.array_equal`.

## Change 1: per-tier instrumentation (plan step 2)

`paper_5/src/paper5/stability/apcm_carried_witness.py`
- `75-110` `_TIER_NAMES` and `_TierClock` (accumulating seconds and call
  counts for the five named tiers).
- `861-872` `CarriedWitnessModel.enable_tier_timing` and the `tier_timing`
  property; `_tier_clock` defaults to `None`, so an uninstrumented rollout
  pays one attribute read per instrumented stage and never calls
  `perf_counter`.
- instrumented stages: `spectral_enclosure`, the predictor
  eigendecompositions in `_solve_critical_schur` and
  `_solve_relaxed_readable_bundle_cone`, the assembly and svec blocks in
  `_solve_critical_schur`, `_solve_relaxed_readable_bundle_cone` and
  `_solve_relaxed_readable_full_cone`, all four live `clarabel.DefaultSolver
  .solve()` sites, `_event_completion_retraction` as a whole, and the
  retained-plus-completion velocity evaluation at the top of `radial_atom`.

`paper_5/src/paper5/stability/apcm_carried_witness_analysis.py`
- `--tier-timing` flag; enabled after `prepare` so the accumulators cover the
  rollout only; `summary["integration"]["tier_seconds"]` carries `seconds`,
  `calls`, `rollout_wall_seconds` and `unaccounted_seconds`, next to the new
  `maximum_atom_seconds` entry.

Not instrumented, because no caller resolves them: `_solve_cut_qp`,
`_solve_relaxed_readable_schur`, `_solve_relaxed_readable_cutting_plane`
(confirmed by grep over `paper_5`, not by the graph alone).

Gates: 22 carried-witness tests pass; both instrumented runs reproduce their
reference arrays bitwise (worst relative difference 0.000e+00).

## Change 2: precompiled commutator structure and hoisted center-cross reads

`paper_5/src/paper5/stability/apcm_carried_witness.py`
- `382-465` `_COMMUTATOR_ROW_TABLES`, `_commutator_row_table`,
  `_commutator_rows`, `_moment_derivative_from_terms`; `_moment_derivative`
  now builds the term list once and delegates.  The compiled rows are
  `(term index, generated moment, commutator coefficient)` in the same
  nested term-then-commutator order, so the accumulation is bitwise the
  same sum.  The table is keyed on the Hamiltonian word tuple, so a changed
  operator basis compiles its own rows.
- `retained_velocity` and `desired_completion_velocity` hoist
  `_hamiltonian_terms`, the word tuple and the table out of their key loops.
- `783-830` `CarriedWitnessGeometry.center_cross_evaluator` (new) forms the core
  overlap product and the two row lookups once per state pair;
  `center_cross_value` is now a one-shot wrapper over it, and
  `desired_completion_velocity` uses the evaluator inside its triple loop.
  `_center_row_lookup` and the new `_extra_row_lookup` are memoized on the
  geometry instance.

Effect (unguarded velocity tier, the event guard's steady state):

| run | tier seconds before | after | factor |
|---|---:|---:|---:|
| no guard, t = 0 to 0.05, 40 atoms | 0.7386 | 0.1144 | 6.5 |
| strict, t = 0 to 0.01, 8 atoms | 0.1487 | 0.0278 | 5.3 |

No-guard rollout 0.9284 s -> 0.3034 s; autonomous wall 1.2765 s -> 0.6392 s.

Gates: 25 carried-witness tests pass, including a new test that compares
`_moment_derivative` against the literal
`_hamiltonian_terms`/`_commutator` double loop with `==` on every readable
frontier key, and one that compares the evaluator against
`center_cross_value` with `==` on all 2 * (9 + extra) center-cross entries.
Both anchors reproduce bitwise (worst relative difference 0.000e+00):
`output/local_runs/paper_v_cwrmf_b4_noguard_t005_h0025_20260914_v1`,
`output/local_runs/paper_v_cwrmf_b4_strict_t001_h0025_20260914_v1`.

## Change 3: vectorized `_clarabel_svec_upper`

`paper_5/src/paper5/stability/apcm_positive_extension.py:177-204`
`_clarabel_svec_upper_pattern` (an `lru_cache` on the matrix size) returns
the column-major upper-triangle row and column indices with their svec
scales; `np.tril_indices(size)` read as `(column, row)` reproduces exactly
the `for column: for row in range(column + 1)` order, and the off-diagonal
factor is the same `np.sqrt(2.0)`.

Bitwise equal to the literal packer on symmetric and nonsymmetric inputs at
sizes 1, 2, 5, 62, 80 and 124 (new repository test).  Microbenchmark: 340
packs of one 80x80 matrix, 0.0998 s literal against 0.0032 s vectorized
(31.5x).

In the strict prefix the coefficient assembly and svec tier fell from
2.4676 s to 0.6267 s (3.9x); the remainder of that tier is the
`pa,jpq,qb->jab` bundle compression, not the packing.  Strict rollout
145.87 s -> 142.55 s, which is 2.3% and inside the run-to-run Clarabel
spread.

Gates: tests pass; both anchors bitwise
(`output/local_runs/paper_v_cwrmf_b1_noguard_t005_h0025_20260914_v1`,
`..._b1_strict_t001_h0025_20260914_v1`).

## Change 4: step-scaled completion coefficient tensor cached per rollout

`paper_5/src/paper5/stability/apcm_carried_witness.py:874-893`
`CarriedWitnessModel._step_scaled_completion_coefficients(step_size)` caches
the (340, 62, 62) complex product on the exact step size and marks it
read-only.  Wired into all five solver-tier entries that previously rebuilt
it (`_solve_critical_schur`, `_solve_relaxed_readable_cutting_plane`,
`_solve_relaxed_readable_bundle_cone`,
`_polish_terminal_completion_velocity`, `_solve_relaxed_readable_full_cone`).

Strict rollout 142.55 s -> 139.90 s; the saving is one 20.9 MB allocation and
multiply per tier entry, and it lowers rather than raises peak resident
memory (ram_guard peak 1980 MB reference, 1504 MB after change 3).

Gates: tests pass, with a new test asserting the cached array equals
`step_size * scaled_completion_coefficients()` elementwise, that a repeated
call returns the same object, and that a changed step recomputes; both
anchors bitwise (`..._b2_noguard_...`, `..._b2_strict_...`).

## Change 5: batched mode-gradient contraction, as a named option (off by default)

The batched contraction is **not** bitwise neutral, so it is a declared
option and not the default.

`paper_5/src/paper5/stability/apcm_carried_witness.py`
- `CWRMFSettings.batched_mode_gradients: bool = False` (`:139`) (strict and balanced
  profiles both inherit the default).
- `302-337` `_mode_eigenvalue_gradients(coefficients, vectors, *, batched)` holds both
  forms: the literal one `a,jab,b->j` contraction per mode, and one
  `ra,jab,rb->rj` contraction for the whole set.
- Used by `_event_completion_retraction` and `negative_mode_coupling`; the
  latter reads the flag through `getattr`, because the fixed-dictionary
  route passes a duck-typed adapter (`projected_apcm.py:682`) that carries no
  CWRMF settings, and that adapter must keep the default form.
- Driver flag `--batched-mode-gradients`.

Measured on the compiled (340, 62, 62) tensor at the cutoff-16 correlated
preparation, gradients of the lowest R eigenvectors of the scaled 62-row
Gram at t = 0:

| working modes R | per-mode seconds | batched seconds | factor | maximum absolute departure | departure relative to the largest gradient entry |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.0030 | 0.0032 | 0.9 | 5.551e-17 | 2.158e-16 |
| 4 | 0.0116 | 0.0029 | 4.0 | 2.220e-16 | 3.707e-16 |
| 12 | 0.0344 | 0.0040 | 8.6 | 2.220e-16 | 2.776e-16 |
| 30 | 0.0863 | 0.0036 | 23.8 | 2.220e-16 | 2.254e-16 |
| 61 | 0.1716 | 0.0054 | 31.5 | 2.220e-16 | 2.254e-16 |

The departure is one unit in the last place of the largest gradient entry,
inside the 1e-13 relative tolerance the task allows, but it is a different
BLAS blocking and the retraction's acceptance band is `event_gram_floor`
= 1e-8 with Lawson--Hanson active-set selection, so leaving it off keeps the
event-triggered path reproducible.  On synthetic random coefficients of the
same shape the departure is larger (6.2e-13 relative for the batched einsum,
1.8e-12 for a flatten-plus-GEMM variant), which is why the number above is
quoted on the real tensor and not on random data.

Both touched functions run only on the event-triggered path, so neither
anchor exercises them; the gate is the new repository test, which checks the
default per-mode form against the literal einsum with `==` for each of 12
modes and the batched form against it within 1e-13 relative.

## Change 6: bounded local step halving in `integrate_cwrmf_ssprk2` (plan step 4)

`paper_5/src/paper5/stability/apcm_carried_witness.py`
- `3400-3542` `_StepAttempt` and `_cwrmf_ssprk2_step`: one SSPRK2 step of a given size
  from a given state, returning the two radial atoms' diagnostics and the
  convex-node verdict.  The guard policy, the node test, and every failure
  message string are the ones the loop used before.
- `integrate_cwrmf_ssprk2(..., maximum_step_subdivisions=0)`: at the default
  zero the behaviour is exactly the recorded one.  At depth `d`, a failed
  radial atom or an unresolved convex node retries the interval from the
  last accepted state as `2**k` steps of `time_step / 2**k` for
  `k = 1 .. d`, and stops at the first `k` that carries the whole interval.
  The reported time grid is unchanged: only the grid endpoint is recorded,
  with the interval's maximum atom seconds, maximum correction iterations,
  maximum readable-rate residual, maximum completion correction norm,
  minimum velocity margin and maximum critical modes.  The ultimate failure
  message carries `at subdivision depth d`; the progress line carries
  `subdivision_depth=d`.
- Driver flag `--maximum-step-subdivisions` (default 0), recorded in
  `summary["integration"]["maximum_step_subdivisions"]`.

Gates: two new tests.  One forces every atom to fail and shows the default
path still terminates at step 0 with the unchanged
`first atom at t=0.000000: ...` message, no subdivision word, and exactly one
atom call at the full step; the other forces failure only above half the
step, shows `maximum_step_subdivisions=0` stops at completed_steps 0 and
`maximum_step_subdivisions=1` completes the step with 5 atom evaluations
(one failed full step plus two accepted half steps).  Both anchors reproduce
bitwise.

## Final state

Suite: `cd paper_5 && PYTHONPATH=src python -m pytest -q -p no:cacheprovider`
gives 319 passed, 9 warnings in 110 s (311 before this work; 8 tests added,
all in `paper_5/tests/test_apcm_carried_witness.py`, which went from 22 to
30).

Anchors, final tree against the pre-change references, every array in both
npz files bitwise equal (worst relative difference 0.000e+00):

| run | before | after | factor | all-coordinate scalar RMS against the exact cutoff-16 reference |
|---|---:|---:|---:|---:|
| no guard, t = 0 to 0.05, h = 0.0025 | 1.2864 s | 0.6314 s | 2.04 | 5.836875e-05 both |
| strict, t = 0 to 0.01, h = 0.0025 | 143.511 s | 139.883 s | 1.026 | 5.827262e-05 both |

Final tier split, same two runs:

| tier | no guard, 40 atoms | strict, 8 atoms |
|---|---:|---:|
| Clarabel solve | 0.000 | 138.644 |
| coefficient assembly and svec packing | 0.000 | 0.571 |
| 62-row eigendecomposition | 0.081 | 0.047 |
| unguarded velocity evaluation | 0.114 | 0.027 |
| rollout | 0.296 | 139.535 |

Artifacts: `output/local_runs/paper_v_cwrmf_final_noguard_t005_h0025_20260914_v1`,
`output/local_runs/paper_v_cwrmf_final_strict_t001_h0025_20260914_v1`,
plus the per-change directories `..._{instr,b4,b1,b2,c}_{noguard,strict}_...`
and the flag check
`output/local_runs/paper_v_cwrmf_flagcheck_noguard_t00125_20260914_v1`
(`--maximum-step-subdivisions 2 --batched-mode-gradients --tier-timing`).

## Kept

- Per-tier instrumentation, off by default, `--tier-timing`.
- Precompiled commutator structure and hoisted center-cross reads (bitwise).
- Vectorized `_clarabel_svec_upper` (bitwise).
- Step-scaled coefficient cache (bitwise).
- Bounded local step halving, default 0 (default path bitwise).
- Batched mode gradients as an off-by-default named option.

## Dropped

- Batching the per-mode eigenvalue-gradient contraction **as the default**.
  It is one ulp away from the per-mode form on the real tensor and up to
  6.2e-13 relative on random data of the same shape, so it fails the bitwise
  contract; kept behind `batched_mode_gradients` instead.
- Speeding up `spectral_enclosure` itself.  It spends most of its time on
  three `np.linalg.norm(..., ord=2)` calls, each a full 62x62 SVD, and the
  Hermitian shortcut (`eigvalsh` max absolute value) is a different
  algorithm, so it would not be bitwise neutral.  It is 0.081 s of the
  0.296 s no-guard rollout and 0.047 s of the strict rollout, so it is worth
  revisiting only as a named option.
- Instrumenting `_solve_cut_qp`, `_solve_relaxed_readable_schur` and
  `_solve_relaxed_readable_cutting_plane`: a grep over `paper_5` finds no
  caller for any of the three.

## Blockers and what the numbers say about the plan

- Plan step 3 ranked the accelerations by code reading and put
  `_clarabel_svec_upper` first.  Measured, the strict guarded atom is 98.1%
  Clarabel solve, so the four accelerations together move the strict prefix
  by 2.5% (143.511 s -> 139.883 s over 4 steps).  Making the strict guard
  fast needs the conic solve itself to change: fewer solves (the event
  policy of plan step 5), a smaller cone, or looser tolerances -- each a
  scientific decision, not a refactor.
- The accelerations do pay on the unguarded atom, which is the
  event-triggered guard's steady state: 2.04x on the no-guard rollout, and
  the unguarded velocity tier itself 6.5x.  At 400 atoms over t = 0 to 0.5
  the record's 18.93 s no-guard line should fall correspondingly; the
  parent session's t = 0.5 anchor will show it.
- Not attempted here: Clarabel `max_threads` (plan step 7 gates it on steps
  5 and 6, and the record says the 2026-08-05 four-thread accelerator failed
  parity).
