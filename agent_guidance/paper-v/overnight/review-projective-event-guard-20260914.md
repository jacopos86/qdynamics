# Independent review of `projective-event-guard`, 2026-09-14

Reviewer agent.  No source file edited.  Worktree
`/Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914`,
branch `claude/paper-v-routes-20260914`.

## Verdict

Pass.  Scope, both cited test commands, default-path parity, the
named-option discipline, and every primary number reported hold under
independent re-execution.  One reported number is wrong and must be
corrected before the worklog entry is written (finding 1); five further
findings are non-blocking.

## Scope

`git status --short` shows eight modified tracked files and two untracked
source/test files.  Four modified plus the two untracked belong to this
implementer, and all lie inside the section 7 allowlist:

- `paper_5/src/paper5/stability/apcm_projective_guard.py`
- `paper_5/src/paper5/stability/projected_apcm.py`
- `paper_5/src/paper5/stability/apcm_projective_event_guard_analysis.py` (new)
- `paper_5/tests/test_apcm_projective_event_guard.py` (new)
- the report under `agent_guidance/paper-v/overnight/`
- run directories under `output/local_runs/paper_v_apcm_projective_*`

`apcm_moment_projection.py` is untouched, as are `tests/test_apcm_projective_guard.py`
and `tests/test_projected_apcm.py`.  The two files the implementer was told
not to edit are modified by other agents, not by this one:
`apcm_carried_witness.py`, `apcm_carried_witness_analysis.py` and
`apcm_positive_extension.py` carry only the `cwrmf-cone-speedup` agent's
fingerprints (`_TierClock`, `tier_seconds`, `_clarabel_svec_upper_pattern`),
and `cone_correction.py` plus `test_cone_correction.py` carry a third
agent's barrier-correction and halfspace-projection work (`+560/-134`,
none of it referencing the projective guard).  `git status --short
--untracked-files=all` returns nothing outside `paper_5/` and
`agent_guidance/paper-v/`.  The main checkout is untouched:
`/Users/lordskyler/Holstein_test/paper_5/src/paper5/stability/{projected_apcm,
apcm_projective_guard,apcm_moment_projection}.py` still carry their
2026-08-31 20:31 mtimes, and neither the new driver nor any
`paper_v_apcm_projective_*` directory exists there.  No scope violation.

## Tests, rerun verbatim

| command | result |
|---|---|
| `cd paper_5 && PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/test_apcm_projective_guard.py tests/test_projected_apcm.py tests/test_apcm_projective_event_guard.py` | 30 passed in 23.95 s |
| `--collect-only` on the two pre-existing files alone | 20 tests collected |
| `cd paper_5 && PYTHONPATH=src python ../pipelines/shell/ram_guard.py --limit-mb 5000 -- python -m pytest -q -p no:cacheprovider tests` | 326 passed, 9 warnings in 107.61 s, peak 2110 MB |

The focused figure matches exactly.  The full-suite figure is 326, not the
reported 319: seven tests were added to `tests/test_cone_correction.py` by a
concurrent agent after this implementer's run.  Zero failures either way, so
the gate holds.

## Default-path parity, re-checked against `HEAD` by the reviewer

The implementer's own bitwise gates compare guard-on against guard-off
inside one process.  That does not establish that today's default path
reproduces the pre-change default path, so I built the comparison myself:
a copy of `paper_5/src/paper5` with only `projected_apcm.py` and
`apcm_projective_guard.py` restored from `HEAD` (both edited files; every
other module left at the worktree's current state, so the concurrent agents'
edits are common to both sides), placed under a scratch root that
`find_repo_root` resolves, and the same script run against each.

Configuration: `DimerParameters(lambda_ep=1.5)`, the declared 15-key
entrance dictionary, phonon cutoff 6, fixed-dictionary APCM SSPRK(3,3) with
containment before every RHS call, t = 0 to 0.01 at h = 0.0025 (4 steps);
plus `select_outer_feasible_projective_guard` on the two-key entrance
fixture at cutoff 6, boxed, in both `canonicalize=False` and
`canonicalize=True`.

Every array compared with `np.array_equal(..., equal_nan=True)`:

| array | bitwise |
|---|---|
| `times`, `states` (5x44), `base_minimum_eigenvalues`, `extension_minimum_eigenvalues` | equal |
| `retained_correction_norms`, `auxiliary_correction_norms`, `stage_retractions`, `accepted_substeps` | equal |
| `rhs_evaluations` / `completed_steps` / `success` | equal |
| guard `target_image`, `current_standardized_values`, objective, `current_minimum_scaled_eigenvalue`, `witness_source`, Clarabel status, both `canonicalize` modes | equal |

`ALL BITWISE EQUAL`.  The pre-change tree was confirmed to contain zero
occurrences of `ProjectiveOuterEventGuard` or `frontier_weight`.

Second, stronger check at the reported configuration: I reran the
fixed-dictionary APCM rollout with `outer_guard=None` at phonon cutoff 16,
lambda_ep = 1.5, gamma = 0.5, drive amplitude 1, t = 0 to 0.05 at
h = 0.0025, under `ram_guard --limit-mb 4000` (44.70 s, peak 297 MB), and
compared it against the implementer's guarded
`autonomous_trajectory.npz`.  `states`, `times`,
`extension_minimum_eigenvalues`, `base_minimum_eigenvalues`,
`stage_retractions` and `accepted_substeps` are all bitwise equal, and
`rhs_evaluations` is 66 on both sides.  Since no stage triggered, the
guarded rollout is the default rollout.

## Defaults untouched, new behaviour behind named options only

Read line by line in `git diff`:

- `_solve_explicit_target_qp(..., frontier_weight: float = 0.0)`
  (`apcm_projective_guard.py:1153`): at 0.0 the frontier block is the same
  `sparse.csc_matrix((count, count))` the recorded path built.
- `select_outer_feasible_projective_guard(..., frontier_weight: float = 0.0)`
  (`2451`): `witness_source` is `"outer_target_stage"` and the
  minimum-norm lift still runs whenever `frontier_weight == 0.0`.
- The two added gating conjuncts (`and frontier_weight == 0.0` on the
  relative-face retry; `canonicalize and frontier_weight == 0.0` on the
  lift) are both true at the default.
- `integrate_projected_apcm_ssprk3(..., outer_guard=None, deadline=None)`:
  with `outer_guard=None` the `contain` closure issues the original
  `model.contain_stage` call with the original arguments; with
  `deadline=None` the slice length `kept` equals `intervals + 1`,
  `completed_steps` is `intervals`, `success` is `True` and `message` is
  `"completed"`, exactly as before.
- `FixedDictionaryProjectedAPCM.contain_stage(..., completion=None)`: at
  `None` the declared proximal log-det completion is computed as before.
- `ProjectedAPCMTrajectory.outer_guard_triggered_stages` is a trailing
  field with default `None`.
- `ProjectiveOuterEventGuardSettings` (`projected_apcm.py:577-610`) carries
  `event_gram_floor` 1e-8, `event_gram_hard_floor` 1e-3,
  `event_coupling_threshold` 1e-2, `event_coupling_horizon` 0.05 — verified
  character-for-character equal to `CWRMFSettings` at
  `apcm_carried_witness.py:118-121`, including the two `__post_init__`
  validations.

## Numbers re-read from the artifacts, and one recomputed from scratch

Every value in the implementer's results list matches its `summary.json`.
Independently recomputed rather than read back:

- All-coordinate scalar RMS error of the 31 closed scalar coordinates
  against the exact cutoff-16 driven reference, over 21 sample times,
  t = 0 to 0.05 at h = 0.0025: I loaded
  `approximate_archive_coordinates` and `exact_archive_coordinates` from
  `trajectory.npz` and computed `sqrt(mean((a-e)**2))` myself =
  **7.931831e-05**, maximum absolute coordinate error 3.819713e-04, initial
  canonical-embedding offset 8.0778e-16 in Euclidean norm.  Matches.
- The comparison against the record's 3.24e-06 is apples-to-apples and
  I confirmed it against a same-day measurement rather than the worklog:
  the plan's step-4.1 strict always-on CWRMF anchor at the identical
  horizon (`output/local_runs/paper_v_cwrmf_strict_t005_h0025_cutoff16_20260914_v1`,
  t = 0 to 0.05 at h = 0.0025, cutoff 16, strict profile) reports
  `dynamic_all_coordinate_scalar_rms_error` 3.2359165867595688e-06.  The
  projective rollout's own dynamic figure equals its raw figure because its
  initial offset is 8.08e-16, so the ratio is 7.931831e-05 / 3.2359166e-06
  = **24.51**.  The implementer's "factor 24.5" and their caveat about the
  offset are both correct.  (The raw `all_coordinate_scalar_rms_error` of
  that CWRMF anchor is 5.8369e-05, carrying a 3.2442e-04 fixed embedding
  offset; comparing against that number instead would have been wrong, and
  the implementer did not.)
- Event-test cost re-derived from `outer_guard_stages.jsonl`: 65 records,
  `trigger_wall_seconds` total 0.013812 s, mean 2.1250e-04 s, maximum
  5.5946e-04 s, which is 0.0273% of the 50.5756 s autonomous wall time;
  `endpoint_unshifted_minimum` ranges -1.3846924140786845e-13 to
  +2.874887e-05; 0 triggered; every `reason` is `quiet`; 7 stage
  retractions.  All match.
- Forced-trigger run v2: 460.06 s for one step, per-stage selector 45.208 /
  45.037 / 364.814 s, outcomes `selector_rejected:explicit_target:NumericalError`
  twice and `...:AlmostSolved` once, peak resident 5293129728 bytes
  (5.29 GB), 0 retractions, `event_gram_floor` 1e-17 and
  `event_gram_hard_floor` 1e-16 recorded in `guard_settings`.  All match.
  The v1 directory holds only `progress.jsonl`, consistent with the
  declared `ram_guard` kill.
- Gate reorder: 46.362 s then 45.963 s, both
  `explicit_target:NumericalError`, objective 0.011286373848805029 in both,
  minimum scaled eigenvalue -2.5903862151141617e-09, speedup 1.00868, peak
  3802923008 bytes.  All match.
- The eight strictly convex points across the two selector-audit artifacts
  match the report value for value.
- Cone dimensions: I constructed
  `projective_guard_outer_extension(SymmetryReducedPositiveExtension(active_keys=ENTRANCE_RELATIVE_MOMENT_KEYS))`
  and confirmed 60 Gram rows and 238 frontier moments on the one-shell
  outer cone, against 24 and 62 on the current cone.
- Code claims spot-checked: the retry trigger set
  `{"AlmostSolved", "InsufficientProgress", "MaxIterations"}` is at
  `apcm_projective_guard.py:1353`; `_EXPLORATORY_ACCEPTANCE_TOLERANCE = 1e-6`
  is at `610` and is applied at `690-700`;
  `SymmetryReducedAPCMGeometry.retract_stage` does re-solve the declared
  completion on the retracted lower moments at
  `apcm_moment_projection.py:1312-1315`, so the `dropped` entry about a
  retracted stage is accurate.

## Findings

1. **A reported number is wrong.**  "All four points above lie within 0.17
   percent of it" (report, section (c)) and "all eight objectives lie
   within 0.35 percent of the record's 0.01126712315" (structured report)
   both understate the spread.  Measured deviations from 0.01126712315:
   w=1e-5 0.0348%, w=1e-4 0.0802%, w=1e-3 0.1380%, w=0 0.1709%, w=1e-2
   0.2112%, w=1e-1 0.2964%, w=1e-6 0.3463%, w=1 0.3738%.  The correct
   statement is "within 0.38 percent", or the range 0.035% to 0.374%.  The
   table entries themselves are all exactly right and the conclusion (the
   selector reaches the same neighbourhood of the minimizer under every
   weight; the open question is certification, not location) is unaffected.
   Fix both sentences before the worklog entry is written.
2. **Full-suite count.**  326 passed today against the reported 319; seven
   tests were added to `tests/test_cone_correction.py` by a concurrent
   agent after the implementer's run.  No failures.
3. **Declared memory deviation, non-blocking.**  The forced-trigger
   measurement ran under `--limit-mb 6000` rather than the 4000 the
   implementer was given, at 5.29 GB peak.  Section 7 of the plan sets
   5000 and plan step 4.1 uses 6000, and the 18 GB aggregate held, so this
   is inside the binding contract; the implementer declared it.
4. **Stale line reference.**  The report and the `ProjectiveOuterEventGuardSettings`
   docstring cite `apcm_carried_witness.py:81-83` for the mirrored floors;
   a concurrent agent's `_TierClock` insertion moved them to `118-121`.
   The names and default values are identical, so only the citation drifted.
5. **Report header still reads "Status: in progress."** despite the
   structured report being final.
6. **Section 7's recording requirement is unmet.**
   `paper_5/notes/electron_phonon_closure_worklog_20260803.md` is
   unmodified: no dated entry for these experiments.  The parent session
   owns that entry; it should carry the corrected percentage from
   finding 1 and the contemporaneous 3.2359166e-06 CWRMF strict figure
   rather than the worklog's 3.24e-06.

## Reproducibility caveat carried forward

The implementer's blocker about the shared worktree stands and is worth
repeating: `apcm_carried_witness.py`, `apcm_carried_witness_analysis.py`,
`apcm_positive_extension.py`, `cone_correction.py` and two test files are
concurrently modified.  `_accuracy_metrics`, which the driver imports for
scoring, is **not** among the functions those agents touched (their diff in
that file is confined to `_parser` and `main`), so the accuracy numbers are
against the `HEAD` scorer.  My parity comparison holds those modules fixed
on both sides, so it isolates this implementer's two files exactly.
