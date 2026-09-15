# review-projective-gap (2026-09-15)

Independent review of the round-2 implementer `projective-gap`.  Branch
`claude/paper-v-routes-20260914`, worktree
`/Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914`,
HEAD `fbf004a3` (the parent's worklog-only commit on top of round 1's
`0e7bb66b`).  No source file was edited by this review; the only paths written
are this report and
`output/local_runs/paper_v_apcm_projective_review_default_recheck_t005_20260915_v1/`.

## Verdict

**Pass.**  Every measured number in the implementer's report reproduces from
the artifacts to the digits printed, the recorded path is bitwise unchanged
under a rollout this review ran itself, the tests rerun at the cited counts,
and nothing outside the section-7 allowlist or the implementer's own files was
touched.  Four reporting defects, three of them wrong counts, must be
corrected before the worklog entry is written; none of them touches a
scientific claim, the code, or a default.

## Scope

`git status --short` lists nine modified and thirteen untracked paths; four
round-2 implementers share this worktree.  Attribution:

| path | owner |
|---|---|
| `paper_5/src/paper5/stability/projected_apcm.py` | projective-gap |
| `paper_5/src/paper5/stability/apcm_projective_guard.py` | projective-gap |
| `paper_5/src/paper5/stability/apcm_projective_event_guard_analysis.py` | projective-gap |
| `paper_5/tests/test_apcm_projective_event_guard.py` | projective-gap |
| `agent_guidance/paper-v/overnight/projective-gap-20260915.md` (new) | projective-gap |
| `output/local_runs/paper_v_apcm_projective_*_20260915_v1/**` | projective-gap |
| `paper_5/src/paper5/stability/apcm_carried_witness.py`, `apcm_carried_witness_analysis.py`, `tests/test_apcm_carried_witness.py`, `cwrmf_solver_instances.py`, `cwrmf_solver_replay.py`, `cwrmf_solver_replay_bench.py`, `tests/cwrmf_solver_*_test.py` | cone-solve-instances |
| `paper_5/src/paper5/stability/matrix_reference.py`, `tests/test_matrix_scalar_parity.py`, `tests/test_matrix_reference_moment_assembly.py` | matrix-reference-speedup |
| `paper_5/src/paper5/stability/cone_long_horizon.py`, `tests/test_cone_long_horizon.py`, `pipelines/open_dynamics/run_paper_v_energy_neutral_barrier_long_horizon.py` | boson-barrier-t100 |

The four files the implementer changed are four of the nine they were given;
`apcm_moment_projection.py` and `adaptive_positive_moment.py` are unchanged,
and `apcm_carried_witness.py`, `cone_correction.py` and `matrix_reference.py`
were not edited by them.  No path outside `paper_5/**`,
`agent_guidance/paper-v/**`, `pipelines/open_dynamics/**` or
`output/local_runs/paper_v_*` was touched.  In the main checkout
`/Users/lordskyler/Holstein_test`, `projected_apcm.py` and
`apcm_projective_guard.py` still carry their 2026-08-31 20:31 mtimes and the
other two files do not exist there at all.  Nothing was committed by the
implementer: `fbf004a3` adds only 64 lines to the worklog.  No scope
violation.

## Defaults and the recorded path

Read of the whole diff: every new behaviour is a named field or keyword whose
default is the committed path.

- `ProjectedAPCMStagePolicy` defaults `stage_retraction=True`,
  `quiet_containment=True`, `completion_start="warm"`,
  `carried_completion=False`.  In `contain_stage` the default policy takes the
  `is_default` branch, which is the original
  `self.extension.complete(lower, warm_frontier=warm_frontier)` statement
  verbatim, and both early returns are guarded by `not policy.quiet_containment`
  and `not policy.stage_retraction`, so neither is reachable at the default.
- `stage_triggered` is read only inside the `quiet_containment` branch, so the
  guard passing it changes nothing at the default.
- `tolerate_stage_failure=False` re-raises `ProjectedAPCMFailure` and
  `APCMProjectionError` unchanged.
- `frontier_metric="identity"` rebuilds the recorded `w I` block, and
  `frontier_weight == 0.0` still short-circuits to the empty block;
  `warm_refinement_passes=0` leaves the single Clarabel call and the original
  `relative_gap`, since `_original_relative_duality_gap` is applied only
  under `if refinement_statuses:`.
- `exploratory_acceptance_tolerance=None` resolves to the module constant
  `_EXPLORATORY_ACCEPTANCE_TOLERANCE = 1e-6`.

Verified, not read: this review reran the recorded-path rollout itself under
the tree as it now stands
(`--mode rollout --final-time 0.05 --time-step 0.0025`, no new option, peak
253 MB under `ram_guard --limit-mb 2000`) into
`output/local_runs/paper_v_apcm_projective_review_default_recheck_t005_20260915_v1/`
and compared it with round 1's
`paper_v_apcm_projective_event_guard_t005_h0025_cutoff16_20260914_v1/`.
All eight arrays of `autonomous_trajectory.npz` -- `times`, `states`,
`approximate_archive_coordinates`, `base_minimum_eigenvalues`,
`extension_minimum_eigenvalues`, `stage_retractions`, `accepted_substeps`,
`outer_guard_triggered_stages` -- are equal under `np.array_equal`, the
all-coordinate scalar RMS is `7.931830738563173e-05` in both, all five
blockwise values are equal as floats, and both report 7 stage retractions.
The implementer's own two baselines
(`ablation_baseline_t005`, `ablation_baseline_recheck_t005`) are also bitwise
equal to round 1 on `states` and `extension_minimum_eigenvalues`.

On the selector side the recorded solve is unchanged in the strongest
available sense: round 1's `strictly_convex_sweep` row at frontier weight
`1e-5` and the round-2 `selector_acceptance2em6_w1em5` solve agree bit for bit
in relative duality gap (`1.1193523667553595e-06`), objective
(`0.01127104613019021`), minimum scaled eigenvalue (`-4.242169565838753e-14`),
stationarity (`5.495497106410521e-08`), feasibility, complementarity and
iteration count (18); only `acceptance` moved from `rejected` to
`provisional_independent_kkt` under the named tolerance.

## Tests rerun

| gate | implementer | this review |
|---|---|---|
| the four owned test files together | 44 passed, 53.67 s | 44 passed, 52.18 s, peak 349 MB under `ram_guard --limit-mb 3000` |
| whole `paper_5` suite | 376 passed, 9 warnings, 139.14 s, peak 1505 MB | 376 passed, 9 warnings, 140.25 s, peak 1526 MB under `ram_guard --limit-mb 3000` |
| `test_apcm_projective_event_guard.py` collected | -- | 20 now, 10 at HEAD |

The two bitwise default-path gates were inspected and do assert what the
report says they assert: `assert_array_equal` on `states`,
`extension_minimum_eigenvalues` and `stage_retractions` of a two-step
`h = 0.0025` rollout with and without the policy spelled out, and array
equality of `target_image`, `current_standardized_values` and the objective
with and without `frontier_metric="identity"`.

## Numbers re-derived from the artifacts

Every value below was recomputed by this review from the run directories, not
copied from the report.

- Baseline, `t = 0` to `0.05`, `h = 0.0025`, 15 entrance keys, cutoff 16:
  all-coordinate scalar RMS `7.931830738563173e-05`; blockwise dynamic scalar
  RMS rho `5.01532e-07`, B `1.59619e-06`, N `2.17278e-04`, A `9.10358e-06`,
  C `2.01524e-05`; 7 stage retractions; minimum unshifted retained joint-Gram
  eigenvalue `2.892080456917745e-05`; 54.04 s.  All as reported.
- The attribution claim, recomputed from `trajectory.npz` and
  `autonomous_trajectory.npz`: the Euclidean norm of the N-block error is
  `2.249966e-08` at `t = 0.0175` with no retraction applied and
  `2.282038e-04` at `t = 0.0200` where one is; at the sample times that apply
  none (`0.0250`, `0.0300`, `0.0350`, `0.0375`, `0.0450`, `0.0475`, `0.0500`)
  it agrees with the previous sample to the four digits the report prints, and
  it steps at every sample that applies one (`4.323964e-04`, `4.893152e-04`,
  `5.635060e-04`, `6.045977e-04`, `6.600439e-04`).  The per-sample retraction
  counts sum to 7.  Confirmed exactly.
- `h = 0.00125`: `1.5805311986340098e-04`, N `4.38778e-04`, rho `2.37961e-07`,
  46 retractions, 132.2 s.  N rises by 2.019 and rho falls by 2.108, as
  reported.
- K/P/D entrance correction off: `2.128060567221998e-03`, rho `1.32353e-04`,
  N `1.14363e-04`, C `3.16496e-03`, 5 retractions, 44.1 s.  `model_settings`
  in the summary records `include_k`, `include_pauli` and
  `include_opposite_spin` all false.
- 13 and 14 entrance keys: `6.2000e-05` / N `1.70733e-04` / 7 retractions and
  `9.2993e-05` / N `2.58117e-04` / 9 retractions, with
  `dictionary.entrance_keys` 13 and 14 against `declared_entrance_keys` 15.
- Both failing ablations: `1.9407597628467683e-06` identically, 0 retractions,
  6 of 20 steps, last time `0.015`, message `moment_extension_failure
  persisted after 8 stage subdivisions`, 23.4 s and 23.3 s.  The recorded path
  restricted by this review to the same seven sample times gives
  `1.9293058827648867e-06` with N `4.257724e-09` against the ablation's
  `4.257683e-09`; both round to the `4.2577e-09` the report prints.
- Carried lift: `3.51369583602504e-05` at `h = 0.0025` (N `8.97755e-05`, 1
  retraction, 41.98 s, 36 accepted of 66 selected), `1.2875122500666576e-05`
  at `h = 0.00125` (N `1.82962e-05`, 113 of 120, 14.18 s),
  `1.4634552226759374e-05` at `h = 0.000625` (236 of 240, 10.12 s).  The
  margin-multiple 0 run carries the same 36 of 66 and reproduces every digit.
  The ratios `6.16`, `3.98` and `3.8` check out.
- The `t = 0.5` carried-lift run: 91 of 400 steps, last time `0.11375`,
  message `extended operator moment matrix numerical face is unstable under
  null-tolerance refinement`, 293 of 300 carried, 1 retraction, 14.80 s,
  all-coordinate scalar RMS `5.8242780016611654e-05`, C `8.50084e-05`, N
  `1.86637e-05`.  Its `times` and `states` arrays over `t = 0` to `0.05` are
  equal under `np.array_equal` to the standalone `h = 0.00125` run and its
  prefix RMS recomputes to `1.2875122500666576e-05`.  The "reproduces exactly"
  claim is exact.
- Cold-start control: `5.109960537812224e-05`, N `1.36831e-04`, 4 retractions,
  86.96 s.
- Selector, one evaluation at the cutoff-16 preparation, `w = 1e-5`, boxed,
  `canonicalize` false: acceptance tolerance `2e-6` gives
  `provisional_independent_kkt`, gap `1.1193523667553595e-06`, objective
  `0.01127104613019021`, minimum scaled eigenvalue `-4.242169565838753e-14`,
  stationarity `5.4955e-08`, restriction to the current 24-row cone
  `-9.795518440180705e-16`, 138.67 s; `readout_diagonal` gives
  `NumericalError`, gap `6.6179e-04`, objective `0.01129774686986394`, minimum
  scaled eigenvalue `-3.3612e-09`, stationarity `1.1869e-04`, 50.15 s; one
  warm refinement pass gives `AlmostSolved+warmAlmostSolved`, gap
  `3.4755e-06`, objective `0.01127107028205352`, minimum scaled eigenvalue
  `-3.9375e-13`, stationarity `1.7207e-07`, 271.72 s.  All as reported.
- The forced-trigger step: `outer_guard_stages.jsonl` shows three stages, all
  `triggered` with reason `hard_floor`, all
  `selector_rejected:explicit_target:NumericalError`, selector wall times
  `46.42`, `43.14`, `58.75` s, `stage_retraction_applied` false on all three,
  step wall time 153.53 s, one-step all-coordinate scalar RMS
  `2.5232307679793967e-07`.  Round 1's own forced trigger at `w = 0` is
  `45.21`, `45.04`, `364.81` s, so the "cost is what the strictly convex
  selector buys" reading is supported.
- The two-key cutoff-6 fixture claim was reproduced by this review by calling
  `select_outer_feasible_projective_guard` on `_small_selector_fixture()` at
  `frontier_weight=1.0`: gap `8.544822030290675e-08` with no refinement and
  `8.088543020798776e-08` with one, matching the report's `8.5448e-08` and
  `8.0885e-08`.
- The "60 Gram rows, 238 frontier moments" description of the one-shell outer
  cone is round 1's own recorded pair, not a new name; the 24 rows and 62
  frontier moments in the summaries are the current cone, which the report
  also names correctly.

Impact analysis spot check from the main checkout's index:
`node .gitnexus/run.cjs impact "_guard_acceptance" --direction upstream`
returns risk `MEDIUM`, 6 direct callers, module `Stability`, matching the
implementer's table.  The index is 9 commits behind this worktree and the
worktree is not indexed, so `detect_changes` is not available here; nothing
was committed, so no commit gate was skipped.

## Defects to correct before the worklog entry

1. **The count of added tests is wrong.**  The report says "eleven tests
   added" and "(20 to 31)", and the structured report says "44 passed ...
   against 34 before this round (11 tests added)".  Ten tests were added:
   `test_apcm_projective_event_guard.py` collects 20 now and 10 at HEAD, and
   the four owned files collect 44 now against 34 before.  `44 - 34 = 10`.
2. **The count of untouched `_guard_acceptance` call sites is wrong.**  The
   report says "the other five callers are untouched" and "unchanged for all
   five other call sites".  There are six call sites in
   `apcm_projective_guard.py` (lines 952, 1166, 1461, 1702, 2118, 2424); two
   of them, `_solve_explicit_target_qp` and
   `_solve_explicit_target_on_relative_face`, now pass the keyword, so four
   are untouched.  The substance -- that no other caller's acceptance rule
   moved -- is correct.
3. **The peak-memory figures carry three different definitions under one
   label.**  In the (c) table the column "peak GB" holds `3.50`, `3.06` and
   `3.46`, which are the summaries' `peak_resident_bytes` divided by 2^30,
   while the round-1 row's `3.84` is round 1's `ram_guard --limit-mb 4000`
   process-tree peak; in the (c) prose, "peak resident 3.90 GB" for the forced
   trigger is `peak_resident_bytes` (3898392576) divided by 10^9, which is
   `3.63` on the table's own convention.  State which measurement and which
   power each number uses, or convert them all to one.
4. **Three cited artifact directories are empty.**
   `..._ablation_keys5_t005_20260915_v1/` and `..._keys12_t005_20260915_v1/`
   contain no files and `..._keys10_t005_20260915_v1/` contains only a
   `progress.jsonl` whose single line is the preparation, so the quoted
   refusal messages (`moment_extension_degeneracy: conic Newton step failed
   descent after relative face support=14/14, directions=11/11`; `required
   relative-mode moment is missing: MomentKey(spin_up='I', spin_down='Y',
   x_power=1, p_power=1)`; `relative face support=20/21, directions=13/48`)
   cannot be recovered from the artifacts.  Either capture the driver's
   stderr into the run directory or say in the report that these three
   messages are transcript-only.

Two smaller notes, not defects: `ProjectedAPCMStagePolicy.is_default` does not
test `carried_completion_margin_multiple`, which is correct because that field
is read only under `carried_completion`, but a later reader should not take
`is_default` to mean "all five fields at their defaults"; and the opening
baseline's `summary.json` predates that field, so its `stage_policy` block has
four keys where the recheck has five, even though the two runs are bitwise
equal.

## Open item

The plan's section 7 requires one dated worklog entry per completed experiment
in `paper_5/notes/electron_phonon_closure_worklog_20260803.md`.  The
implementer did not write one, naming the file as another agent's working copy
this round; the parent session owns the entry.  That entry is still to be
written, with the four corrections above folded in.
