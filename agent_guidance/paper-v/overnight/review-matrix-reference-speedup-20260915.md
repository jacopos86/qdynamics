# review-matrix-reference-speedup (2026-09-15)

Independent review of the round-2 implementer `matrix-reference-speedup`.
Branch `claude/paper-v-routes-20260914`, worktree
`/Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914`,
pre-change reference HEAD `0e7bb66b`.  No source file was edited by this
review; the only file written is this report and the runs under
`output/local_runs/paper_v_matrixref_review_20260915/`.

## Verdict

**Pass.**  All four named gates hold and two of them were reproduced from
scratch by this review against the HEAD build, not read out of the
implementer's artifacts.  Two numbers in the implementer's report are wrong
and must be corrected before the worklog entry is written (below).

## Scope

`git status --short` lists ten modified and thirteen untracked paths in this
worktree; three round-2 implementers are working in it at once.  Attribution:

| path | owner |
|---|---|
| `paper_5/src/paper5/stability/matrix_reference.py` | matrix-reference-speedup |
| `paper_5/tests/test_matrix_scalar_parity.py` | matrix-reference-speedup |
| `paper_5/tests/test_matrix_reference_moment_assembly.py` (new) | matrix-reference-speedup |
| `agent_guidance/paper-v/overnight/matrix-reference-speedup-20260915.md` (new) | matrix-reference-speedup |
| `output/local_runs/paper_v_matrixref_20260915_v1/**` | matrix-reference-speedup |
| `paper_5/src/paper5/stability/apcm_carried_witness.py`, `apcm_carried_witness_analysis.py`, `tests/test_apcm_carried_witness.py`, `cwrmf_solver_instances.py`, `cwrmf_solver_replay.py`, `cwrmf_solver_replay_bench.py`, `tests/cwrmf_solver_*_test.py` | cone-solve-instances |
| `paper_5/src/paper5/stability/apcm_projective_guard.py`, `apcm_projective_event_guard_analysis.py`, `projected_apcm.py`, `tests/test_apcm_projective_event_guard.py` | projective-gap |
| `paper_5/src/paper5/stability/cone_long_horizon.py`, `tests/test_cone_long_horizon.py`, `pipelines/open_dynamics/run_paper_v_energy_neutral_barrier_long_horizon.py` | boson-barrier-t100 |
| `paper_5/notes/electron_phonon_closure_worklog_20260803.md` | event-guard run log (bounded-depth event-triggered guard, first attempt) |

No path outside the section-7 allowlist was touched, `cone_correction.py` was
not touched, and no file under `/Users/lordskyler/Holstein_test` or any
Paper I--IV path has an mtime after 2026-09-14 22:00.  No scope violation.

## Defaults

The whole diff to `matrix_reference.py` is assembly rewriting.  No function
signature gains a parameter, no module constant that carries a tolerance,
threshold, profile or acceptance rule is added or changed, and `strict` and
`balanced` live in `apcm_carried_witness.py`, which this implementer did not
touch.  There is no new option to leave off, which is admissible here only
because the acceptance evidence is byte identity rather than a tolerance --
and that identity is what the gates below establish.

## Gates re-run by this review

1. **Pre-change build is the real pre-change build.**
   `output/local_runs/paper_v_matrixref_20260915_v1/matrix_reference_before.py`
   is byte-identical to `git show HEAD:paper_5/src/paper5/stability/matrix_reference.py`,
   and `matrix_reference_after.py` is byte-identical to the current worktree
   file.  The comparison is therefore between HEAD and what the parent will
   commit.

2. **Gate 1 reproduced from scratch.**  The ten-coordinate energy-neutral
   matrix barrier (`structured_boson_barrier_correction` through
   `closed_cone_projected_rhs`, `lambda_ep = 1.5`, `gamma = 0.5`, drive
   amplitude 1, exact ground-state contraction at phonon cutoff 16,
   `activation_margin = 1e-5`, `target_flux = 0`, `barrier_rate = 5`, DOP853
   rtol 1e-9 atol 1e-11 maximum step 0.05, t = 0 to 2, sampled every 0.05) was
   run by this review on the post-change build into
   `output/local_runs/paper_v_matrixref_review_20260915/gate1_reviewer`:
   all six stored arrays (`times`, `states`, `boson_moment_minima`,
   `electron_minima`, `energies`, `initial_state`) are `np.array_equal` to
   `gate1_before`, and every `summary.json` entry except `wall_seconds` is
   identical, including the working notes' pinned minimum bosonic moment
   eigenvalue `2.893142825017192e-05` and the 749 right-hand-side evaluations
   / 484 half-space projections.

3. **Gate 2 reproduced from scratch.**  The 31-coordinate joint-Gram
   energy-neutral barrier (`structured_electron_phonon_barrier_correction`
   through `closed_electron_phonon_cone_projected_rhs`, same settings plus
   `preserve_correlation_trace=True`, `cone_tolerance = 1e-8`,
   `projection_tolerance = 1e-12`, euclidean correction metric,
   `subtract_initial_residual=False`, t = 0 to 0.05, sampled every 0.0025) into
   `.../gate2_reviewer`: all six arrays `np.array_equal` to `gate2_before`,
   every `summary.json` entry except `wall_seconds` identical, minimum bosonic
   moment eigenvalue `2.4870900303620437e-05`, 224 right-hand-side evaluations,
   1,843 half-space projections, whole `SOLVER_COUNTERS` census identical.
   Integration wall time in this review's single run: `0.5915 s` against the
   implementer's `before` median of `0.92371 s` over seven repetitions.

4. **Gate 3 re-checked array by array.**  `gate3_after` against
   `output/local_runs/paper_v_cwrmf_final_noguard_t005_h0025_20260914_v1`:
   every array of `autonomous_trajectory.npz` and `trajectory.npz` is
   `np.array_equal`, except `velocity_margins` (NaN entries; maximum absolute
   difference 0.0, equal under `equal_nan=True`) and `maximum_atom_seconds`
   (a per-atom wall-clock series).  `accuracy` and `feasibility` compare equal
   as whole dictionaries: all-coordinate scalar RMS error against the exact
   cutoff-16 reference `5.836875347948307e-05`, minimum carried unshifted
   62-row Gram eigenvalue `-8.290500630490342e-05`, over t = 0 to 0.05 at
   h = 0.0025 in the strict numerical profile with `--no-gram-guard`.  All
   settings blocks of `summary.json` are equal; only `integration` differs
   (wall seconds, peak resident, and `tier_seconds`, which is `None` in the
   new run because `--tier-timing` was not passed).  The driver does consume
   `matrix_reference` (`apcm_carried_witness_analysis.py:27-30`, quoted by the
   implementer as `:24-26` before the concurrent cone-solve-instances agent
   inserted an import above it; `apcm_carried_witness.py:57`), so the gate was
   correctly run rather than skipped.

5. **Tests re-run.**  `tests/test_matrix_scalar_parity.py
   tests/test_cone_correction.py tests/test_exact_driven_reference.py
   tests/test_matrix_reference_moment_assembly.py`: **61 passed** in 8.87 s.
   Full `paper_5` suite under `ram_guard --limit-mb 3000`: **376 passed** in
   149.35 s, peak 1,010 MB.  Both figures match the implementer's.

6. **Impact analysis re-run** read-only from the main checkout's index for
   `_validate_matrix_state` (HIGH, 155 impacted, 14 direct),
   `electron_phonon_moment_matrix` (HIGH, 62, 16) and
   `_boson_moment_derivative_from_blocks` (LOW, 13, 4).  Each reproduces the
   implementer's row exactly, and each returned `epistemic: "exact"`, so no
   `UNKNOWN` needed a grep confirmation.

7. **Profile split re-checked** from `profile_before/profile.json` and
   `profile_after/profile.json`: total 1.5590 s to 0.9749 s, function calls
   4,437,442 to 2,083,870, own time in `matrix_reference.py` 0.31535 s to
   0.16041 s, `electron_phonon_moment_derivative` cumulative 0.714 s to
   0.145 s over 7,168 calls in both.  Timing JSON: median of seven
   repetitions 0.9237143330101389 s to 0.5987720839912072 s (35.2 percent
   lower), minimum 0.9071361249953043 s to 0.5964287079987116 s, and the
   harness's own `state_checksum` 148.50903290514307 in both builds.

## Corrections required to the implementer's report

1. **`ndarray.trace` calls do not fall to 0.**  This review profiled the
   post-change build over the identical gate-2 route and counted **448**
   `ndarray.trace` calls remaining (`numpy.trace` wrapper 448,
   `_trace_dispatcher` 448), not 0.  The arithmetic closes exactly: 155,008
   before = 154,560 removed (15 per `_electron_fluctuation_moment_derivative`
   over 7,168 calls, 12 per `_electron_fluctuation_moment_matrix` over 224
   calls, 6 per `_correlation_pauli_block` over 7,392 calls) + 448 that the
   implementer's own "Dropped" section says were deliberately left in
   `matrix_total_energy`, `matrix_state_to_closed_scalar_coordinates` and the
   embedding-residual diagnostics.  The report and the structured summary are
   internally contradictory on this point.  Correct figure: **155,008 to 448
   over 224 right-hand-side evaluations of the joint-Gram energy-neutral
   barrier, t = 0 to 0.05**.  The `np.block` figure is right: `_block` and
   `_block_check_depths_match` are absent from the post-change profile
   entirely, 14,784 to 0, as is the `np.vstack` figure, 9,235 to 1,843.

2. **The boson-block row mislabels the "after" column.**  The report's row
   "the boson block of the moment derivative, cumulative seconds over 7,168
   calls | 0.123 | 0.008" compares a cumulative time with an own time:
   `_fill_boson_moment_derivative` has tottime 0.0082 s and **cumtime
   0.0124 s**, against `_boson_moment_derivative_from_blocks` cumtime
   0.1225 s before.  Cumulative to cumulative the row is 0.123 s to 0.012 s.

Neither correction touches a gate, and neither changes the kept/dropped
verdict.

## Observations, not defects

- `_boson_moment_derivative_from_blocks` now returns `complex128` for every
  input, where `np.block` previously returned `float64` when both blocks were
  real.  Every in-repository caller already passes complex arrays
  (`_phonon_rhs_terms` allocates `dtype=complex`,
  `_closed_scalar_boson_derivative_blocks` returns complex, the zero block is
  `np.zeros((2, 2), dtype=complex)`, and `matrix_dimer_rhs` is complex), and
  the values are equal either way, so nothing in this repository changes.  It
  is a widening a future real-valued caller would inherit.
- `_PAULI_MEAN_CACHE` is module-global mutable state in a scientific module.
  It is keyed on `tobytes()` of a `complex128` `(2, 2)` array, so a hit
  returns exactly what the uncached path would build; the cached array is
  marked read-only and both consumers only read it; the bound of 8 makes it
  FIFO rather than LRU, which can only cost a miss.  Measured on the gate-2
  route: 14,560 calls, 748 of which computed (`_pauli_traces` call count),
  i.e. 13,812 repeats avoided.
- The monomial argument is sound and is what makes the rewrite an identity
  rather than an approximation: for a monomial `M`, `(A @ M)[i, i]` is one
  product plus an exact zero, multiplication by a unit entry of
  `{1, -1, i, -i}` is a component permutation with sign changes, so
  `(D @ sigma_a) @ sigma_b` and `D @ (sigma_a @ sigma_b)` agree in every bit.
  The only residue is the sign of a zero, which `np.array_equal` does not
  separate and which no downstream comparison in this route reads.
- `tests/test_matrix_reference_moment_assembly.py` transcribes the HEAD
  assembly verbatim as `recorded_*`; this review checked each transcription
  against `git show HEAD:...matrix_reference.py` line by line.  The added test
  in `test_matrix_scalar_parity.py` pins the same identity at the
  strong-coupling exact cutoff-16 ground state along the closed-state velocity
  and the 31 closed-state unit directions, which is the operand set the barrier
  runs evaluate.

## Artifacts written by this review

`output/local_runs/paper_v_matrixref_review_20260915/gate1_reviewer/` and
`gate2_reviewer/` (post-change build, compared against the implementer's
`gate1_before` and `gate2_before`).  Both runs went through
`pipelines/shell/ram_guard.py --limit-mb 3000`; recorded peaks 3 MB and 2 MB.
The full suite ran under the same guard at 1,010 MB peak.
