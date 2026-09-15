# Independent review of `cone-solve-instances`, 2026-09-15

Reviewer for the round-2 implementer `cone-solve-instances`.  Worktree
`/Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914`,
branch `claude/paper-v-routes-20260914`, HEAD `7487e861`.  No source file was
edited by this review; only this report was written.

## Verdict

**Fail**, on one claim, not on the work.  The deliverables (a), (b) and (c)
are real, the scope is clean, the tests reproduce, the parity evidence
reproduces to every printed digit, and the default path is unchanged and
provably so.  What fails is a stated finding that the implementer's own
`bench.json` contradicts: the warm-started SCS candidate is reported as
satisfying the balanced acceptance test "on none of the 36", and the
artifact records 15 of 36 under exactly the test every other balanced number
in the report uses.  The argument offered for "none" is a non-sequitur about
a gate that fires unconditionally.  This is a report correction plus, if the
number is wanted, one re-measurement; it does not touch the headline
negative result.

## Gate 1: scope -- passes

`git status --short` and `git diff --stat` against `7487e861`.  Every
modified and untracked path lies inside the section-7 allowlist
(`paper_5/**`, `agent_guidance/paper-v/**`, `pipelines/open_dynamics/**`,
`output/local_runs/paper_v_*`).  Nothing under the main checkout, `src/`,
`test/`, or a Paper I--IV lane.

| file | owner |
|---|---|
| `paper_5/src/paper5/stability/apcm_carried_witness.py` | `cone-solve-instances` (owned) |
| `paper_5/src/paper5/stability/apcm_carried_witness_analysis.py` | `cone-solve-instances` (owned) |
| `paper_5/tests/test_apcm_carried_witness.py` | `cone-solve-instances` (owned) |
| `paper_5/src/paper5/stability/cwrmf_solver_instances.py` (new) | `cone-solve-instances` (owned prefix `cwrmf_solver_`) |
| `paper_5/src/paper5/stability/cwrmf_solver_replay.py` (new) | `cone-solve-instances` (owned prefix) |
| `paper_5/src/paper5/stability/cwrmf_solver_replay_bench.py` (new) | `cone-solve-instances` (owned prefix) |
| `paper_5/tests/cwrmf_solver_instances_test.py` (new) | `cone-solve-instances` (owned prefix) |
| `paper_5/tests/cwrmf_solver_replay_test.py` (new) | `cone-solve-instances` (owned prefix) |
| `paper_5/src/paper5/stability/apcm_projective_guard.py` | `projective-gap` |
| `paper_5/src/paper5/stability/apcm_projective_event_guard_analysis.py` | `projective-gap` |
| `paper_5/src/paper5/stability/projected_apcm.py` | `projective-gap` |
| `paper_5/tests/test_apcm_projective_event_guard.py` | `projective-gap` |
| `paper_5/src/paper5/stability/matrix_reference.py` | `matrix-reference-speedup` |
| `paper_5/tests/test_matrix_scalar_parity.py` | `matrix-reference-speedup` |
| `paper_5/src/paper5/stability/cone_long_horizon.py` (new) | `boson-barrier-t100` |
| `paper_5/tests/test_cone_long_horizon.py` (new) | `boson-barrier-t100` |
| `pipelines/open_dynamics/run_paper_v_energy_neutral_barrier_long_horizon.py` (new) | `boson-barrier-t100` |

No scope violation by this implementer.  `apcm_positive_extension.py` is in
their ownership list and was not touched, which is fine.

## Gate 2: tests -- passes, reproduced exactly

- The three owned files: `49 passed in 2.70 s`
  (`test_apcm_carried_witness.py` 33, `cwrmf_solver_instances_test.py` 5,
  `cwrmf_solver_replay_test.py` 11).  Claimed 49 in 2.87 s.
- Full `paper_5` suite under
  `ram_guard.py --limit-mb 4000`: `377 passed, 9 warnings in 129.85 s`,
  `[ram_guard] exit=0 peak=1705MB limit=4000MB`.  Claimed 377 passed, 9
  warnings, 129.49 s.

The three default-path gates in `test_apcm_carried_witness.py` were read and
are real, not nominal: `test_default_solver_setting_profile_touches_no_clarabel_field`
compares every field of `RECORDED_SETTING_FIELDS` before and after the
overlay; `test_solver_setting_profile_defaults_to_the_pinned_single_thread_set`
asserts both `cwrmf_settings_for_profile("strict")` and `("balanced")` carry
the default and that an unknown name raises;
`test_named_solver_setting_profiles_overlay_only_their_own_fields` pins the
profile table to exactly three entries and their contents.
`test_recorded_atom_reproduces_the_default_path_bitwise` runs the same radial
atom twice on a state displaced into the guard and asserts `np.array_equal`
on the endpoint, the archive velocity, the completion velocity and the
desired completion velocity with and without a recorder installed.

## Gate 3: parity -- passes, re-derived from the arrays

I loaded `autonomous_trajectory.npz` and `trajectory.npz` from the anchor
`output/local_runs/paper_v_cwrmf_final_strict_t001_h0025_20260914_v1` and
from each rerun and compared array by array with `np.array_equal` and a
maximum absolute difference.  Configuration throughout: strict numerical
profile with the always-on guard, `t=0` to `0.01` at `h=0.0025`, 4 SSPRK2
steps, 8 radial atoms, strong coupling `lambda=1.5`, `gamma=0.5`, drive
amplitude 1, phonon cutoff 16, exact-correlated preparation.

- Recording rerun
  (`..._record_strict_t001_h0025_20260915_v1`): every array of both files
  except `maximum_atom_seconds` equal under `np.array_equal`, maximum
  absolute difference `0.000e+00` on all 18.  `summary` reports
  `recorded_conic_instances: 36` and
  `all_coordinate_scalar_rms_error: 5.827261740618364e-05`, identical to the
  anchor's.  Claim confirmed.
- `presolve_off` rerun: identical result, all 18 arrays bitwise, same scalar
  RMS `5.827261740618364e-05`.  Claim confirmed.
- `max_threads_8` rerun: every per-array maximum in the report reproduces
  exactly -- `carried_states` `3.800246e-06`,
  `approximate_archive_coordinates` `2.088607e-15`,
  `minimum_unshifted_eigenvalues` `2.761038e-12`,
  `minimum_shifted_lower_bounds` `2.762656e-12`,
  `readable_rate_residuals` `2.860611e-06`,
  `completion_correction_norms` `7.248094e-06`, `velocity_margins`
  `1.940012e-05`, `critical_modes` 1; `times`,
  `exact_archive_coordinates` and `correction_iterations` bitwise.  Scalar
  RMS `5.827261740618476e-05`.  Claim confirmed.

Wall times and resources read back from the `summary.json` files:
autonomous wall `139.883 / 148.272 / 135.880 / 92.418 s` and Clarabel tier
`138.644 (36) / 146.807 (36) / 134.656 (36) / 91.205 (37) s` for
anchor / recording / `presolve_off` / `max_threads_8`.  All as reported.

The 36 recorded instances reproduce from
`conic_instances/instances.json`: 33 `bundle_cone` and 3 `critical_schur`,
30 `Solved` and 6 `AlmostSolved`, `sum(recorded_clarabel_solve_time) =
145.819 s`, and the per-atom table (side dimension `12 -> 52 -> 62 -> 70 ->
72 -> 76 -> 78 -> 80`, per-atom solve-time sums `0.02, 11.24, 19.51, 26.87,
15.61, 27.30, 19.92, 25.35 s`) matches row for row.

The bench tables reproduce from
`paper_v_cwrmf_replay_bench_strict_t001_20260915_v1/bench.json` and `_v2`:
every total, mean, maximum, acceptance count, agreement count, bitwise
count, maximum velocity difference, maximum cone violation and minimum Gram
eigenvalue in both tables matches the artifact, and every speedup is the
correct ratio against that pass's own `recorded` (`148.407 s` in v1,
`135.669 s` in v2; the v2 bracket is `135.669` and `135.746 s`, spread
`0.077 s`).

## Gate 4: default path -- passes

The default path is unchanged by construction and by measurement.

- `_apply_solver_setting_profile` returns before touching a field when the
  name is `pinned_single_thread`, which is the dataclass default and what
  both `cwrmf_settings_for_profile("strict")` and `("balanced")` produce.
- Every recorder hook is `if self._instance_recorder is not None:`, and
  `_instance_recorder` is initialized to `None`; the driver installs one
  only under `--record-conic-instances`.
- The operand hoisting at the four call sites (`objective_matrix`,
  `objective_vector`, and at the critical-Schur site `assembled_cone_matrix`
  and `assembled_cone_rhs`) builds the same objects in the same order and
  passes them to the same constructor.
- Two full rollouts on the current tree reproduce the round-1 anchor
  trajectory bitwise, one of them with the recorder installed.

Strict and balanced acceptance constants are untouched:
`cwrmf_settings_for_profile` is unchanged in the diff, and the strict
`(schur_safety_margin 2e-12, solver_tolerance 1e-10, psd_inflation 1e-11)`
and balanced `(1e-10, 1e-8, 1e-8)` values are the ones the bench reads.

One measurement-only side effect, not a correctness issue: at the
critical-Schur site the operand assembly moved from inside the
`clarabel_solve` timed region to outside it, so that tier's seconds are not
strictly comparable across rounds.  It covers 3 of 36 calls totalling
`0.018 s`, so nothing reported depends on it.

## Gate 5: reported numbers and vocabulary -- one failure, two errors

### Issue 1 (material).  The SCS balanced-acceptance claim contradicts `bench.json`

The report's prose, its `dropped` entry and the structured summary all say
the warm-started SCS answer "satisfies the balanced acceptance test on none
of the 36".  The artifact says otherwise:
`bench.json -> candidates -> scs_warm_1e_9 -> summary ->
balanced_shifted_accepted = 15`, and the report's own bench-table row for
`scs_warm_1e_9` prints `15` in the balanced-accept column.  `15` is the
count under the substituted-inflation balanced test -- the one the report
declares it uses for every balanced number ("`balanced` throughout is the
substituted-inflation form").  So the table and the prose disagree.

The argument offered for "none" does not support it.  In
`cwrmf_solver_replay.py`, `WarmStartedScsCandidate.solve` calls
`acceptance(..., profile=self.polish_profile, ...)` and leaves
`inflation_shift` at its default `0.0`, so the polish gate is the
*unsubstituted* balanced test.  That test accepts nothing on these operands:
`recorded_balanced_accepted = 0` and `balanced_accepted = 0` for *every one*
of the 22 candidates, Clarabel's own recorded solutions included.  The
polish therefore fires 36 of 36 by construction, and "the polish fired on
all 36" carries no information about the SCS answer.  I confirmed the firing
count directly from the per-instance statuses: 36 of 36 rows of
`scs_warm_1e_9_polish_balanced` carry `->clarabel:` (against 32 of 36 for
`scs_warm_1e_9_polish_strict`, whose gate is the strict test on a
strict-inflation Gram and is therefore meaningful).

Correction needed, either of:

- restate it as measured: under the substituted-inflation balanced test SCS
  is accepted on 15 of the 36 recorded instances and agrees with the
  recorded verdict on 20 of 36; under the unsubstituted balanced test it is
  accepted on 0 of 36, as is Clarabel's own answer, so that test separates
  nothing on strict-profile operands; or
- pass `inflation_shift=BALANCED_INFLATION_SHIFT` into the polish gate and
  re-measure `scs_warm_1e_9_polish_balanced`.

The drop verdict itself is unaffected: SCS is `768.360 s` against
`148.407 s` on the same 36 instances, so it loses on time whatever the
acceptance count.

### Issue 2 (factual).  The bundle iteration range is 14 to 28, not 17 to 28

The report says "Clarabel's iteration count stays between 17 and 28 on every
bundle call" and the structured summary repeats "a flat 17-28 interior-point
iterations".  Over the 33 `bundle_cone` instances the recorded values are
`[14, 17, 17, 17, 18, 18, 18, 19, 20, 20, 21, 21, 21, 22 x 8, 23, 23, 24 x 4,
25 x 4, 28, 28]`: the minimum is 14.  (The three `critical_schur` instances
are 12 each.)  The point the range is making -- that the iteration count is
flat while the solve time rises from `0.006 s` to `14.462 s` -- survives
unchanged.

### Issue 3 (factual).  The recorder write is not inside the Clarabel tier

The parity table explains the recording rerun's `8.4 s` as "the 242 MB of
instance files written inline ... it is in the Clarabel tier because the
write happens between the solve and the next `perf_counter` read".  At all
four call sites the order is `clock.add("clarabel_solve", solve_started)`
*then* `self._instance_recorder.record(...)`, so the write lands outside
that tier.  The measurements agree: `unaccounted_seconds` rose only from
`0.246` to `0.398 s` between anchor and recording rerun, while Clarabel's
own reported `solve_time` summed to `145.819 s` in the recording run against
a `138.644 s` anchor tier.  The `8.16 s` is contention inside the solver,
not the instance write.  This strengthens the parity claim rather than
weakening it, but the sentence as written is wrong about the code.

### Issue 4 (cosmetic).  "ram_guard peak"

The rows labelled `ram_guard peak` (`1609 / 1859 / 1971 / 2443 MB`) are
`summary["integration"]["peak_resident_bytes"]`, the driver process's own
peak resident set from `resource.getrusage(RUSAGE_SELF)`, not a peak
reported by `ram_guard.py` over the process tree.  The values are correct
for the field they come from; the label names a different instrument.

### Vocabulary

No invented quantity name.  `verdict agreement`, `enlargement iterations`
and `the machine's own spread` are each stated as a full sentence in the
record's vocabulary before the short label is used, which is what the naming
rule allows.  `positive-semidefinite cone side dimension` is defined as
twice the spectral bundle rank.  Every table carries its configuration and
horizon in the surrounding text.

## What holds, unqualified

- (a) The recorder is off by default, wired at all four Clarabel call sites
  plus `radial_atom`, and the strict rerun with it on reproduces the round-1
  anchor trajectory bitwise.  36 instances, 33 bundle and 3 critical Schur.
- (b) The bench measures all six candidate groups the task named, on all six
  per-instance quantities the task named, with the strict and balanced
  acceptance tests evaluated afterwards, and the second pass brackets the
  machine spread before drawing a conclusion from a time difference.
- (c) Two named options on `CWRMFSettings.solver_setting_profile` and
  `--solver-setting-profile`, default `pinned_single_thread` applying no
  override, with the strict rollout rerun under each and both compared to
  the anchor on wall time, scalar RMS and maximum coordinate difference.
- The headline negative result -- no candidate that passes the strict
  full-Gram acceptance test on all 36 instances is faster than the pinned
  single-threaded Clarabel call beyond the `0.077 s` machine spread -- is
  supported by the artifacts as written.
