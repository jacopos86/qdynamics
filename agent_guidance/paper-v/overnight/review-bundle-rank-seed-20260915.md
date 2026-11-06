# Review of `bundle-rank-seed`, 2026-09-15

Independent reviewer for the round-3 implementer key `bundle-rank-seed`.
Worktree `/Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914`,
branch `claude/paper-v-routes-20260914`, HEAD at review time `44b97966`
(the implementer's report names `d189c50a`; the parent committed
`44b97966`, the bounded-depth event-guard block, in between, and it does not
touch this key's files).  No source file was edited by this review.

**Verdict: pass.**  Four reporting-level corrections below, none of which
changes a conclusion, a kept option, or the recommendation.

## Scope

`git status --short --untracked-files=all` in the worktree shows twelve
modified files and seven untracked ones.  Attribution:

| file | owner |
|---|---|
| `paper_5/src/paper5/stability/apcm_carried_witness.py` | `bundle-rank-seed` (owned) |
| `paper_5/src/paper5/stability/apcm_carried_witness_analysis.py` | `bundle-rank-seed` (owned) |
| `paper_5/tests/test_apcm_carried_witness.py` | `bundle-rank-seed` (owned) |
| `agent_guidance/paper-v/overnight/bundle-rank-seed-20260915.md` | `bundle-rank-seed` (owned, new) |
| `paper_5/src/paper5/stability/apcm_moment_projection.py`, `apcm_projective_event_guard_analysis.py`, `projected_apcm.py`, `apcm_outer_selector_formulations.py` (new) and the tests `test_apcm_moment_projection.py`, `test_apcm_projective_event_guard.py`, `test_projected_apcm.py`, `test_apcm_outer_selector_formulations.py` (new) | other round-3 keys (`outer-selector-formulation`, `projective-face`) |
| `paper_5/src/paper5/stability/cone_long_horizon.py`, `paper_5/tests/test_cone_long_horizon.py`, `pipelines/open_dynamics/run_paper_v_energy_neutral_barrier_long_horizon.py` | other round-3 key (`joint-gram-t20`) |
| the other untracked `agent_guidance/paper-v/overnight/*.md` | the keys named in their filenames |

No scope violation.  Every modified path lies inside the section-7 allowlist,
nothing under `/Users/lordskyler/Holstein_test` or a Paper I--IV path was
touched, and
`grep -rn "bundle_seed\|bundle_growth\|bundle_interior_overshoot\|_bundle_accepted" --include="*.py" paper_5 pipelines`
returns hits in exactly the three owned files (45, 13, 66 lines), so this key
left no fingerprint on another key's file.  The three files the report lists
as owned-but-untouched (`apcm_positive_extension.py`,
`cwrmf_solver_instances.py`, `cwrmf_solver_replay.py`,
`cwrmf_solver_replay_bench.py` and their two tests) are indeed unmodified.

## Default path unchanged

Read of the diff:

- `_bundle_seed_working_set` under `monotone_hint` returns
  `predictor_vectors[:, np.arange(min(critical_mode_limit, max(spectral_rank,
  self._bundle_rank_hint)))]`, which is line for line the construction the
  diff deletes from `_solve_relaxed_readable_bundle_cone`.  The local name
  `selected` that the old code overwrote is not read anywhere later in the
  method (`awk` over lines 2424--2760 finds it only at its own four lines), so
  dropping that reassignment changes nothing.
- `compressed_margin = schur_safety_margin * bundle_interior_overshoot` with
  the default `1.0` is `schur_safety_margin` exactly; the acceptance test
  still reads `self.settings.schur_safety_margin` and
  `10.0 * self.settings.solver_tolerance` directly.
- the growth padding branch is guarded by `bundle_growth_factor > 1.0`, whose
  default is `1.0`.
- `_bundle_accepted_rank` and `_bundle_accepted_basis` are written on
  acceptance and read only by the non-default policies.
- the six driver flags default to `monotone_hint` / `None` and each `replace`
  fires only when the caller passed the flag.

Measured: I ran the strict `t=0` to `0.01` rollout at `h=0.0025` myself from
the current working tree (strict numerical profile, always-on guard, 4 SSPRK2
steps, 8 radial atoms, strong coupling `lambda=1.5`, `gamma=0.5`, drive
amplitude 1, phonon cutoff 16, exact-correlated preparation, `--tier-timing`,
under `ram_guard --limit-mb 4000`, peak `2047 MB`).  Artifact
`output/local_runs/paper_v_review_bundleseed_default_strict_t001_h0025_20260915_v1/`.
Against the round-1 anchor
`output/local_runs/paper_v_cwrmf_final_strict_t001_h0025_20260914_v1`: all 11
arrays of `autonomous_trajectory.npz` and all 8 of `trajectory.npz` equal
under `np.array_equal` except `maximum_atom_seconds`; all-coordinate scalar
RMS against the exact cutoff-16 reference `5.827261740618364e-05`, identical
to the anchor; 36 `clarabel_solve` calls; Clarabel tier `150.103 s`, inside
the implementer's declared four-bracket spread `150.703` -- `152.774 s`; every
reported feasibility quantity identical to the anchor's
(`maximum_readable_rate_residual 1.4395728609345863e-07`,
`maximum_completion_correction_norm 2.953556872347944e-06`,
`minimum_carried_unshifted_eigenvalue -3.8881605143449574e-12`).

I also re-checked the implementer's own parity claims array by array: the four
default brackets (`..._seed_default_*`) and the `previous_accepted_rank` run
each differ from the anchor in `maximum_atom_seconds` alone.

## Tests rerun

- the three owned files, `cd paper_5 && PYTHONPATH=src python -m pytest -q -p
  no:cacheprovider tests/test_apcm_carried_witness.py
  tests/cwrmf_solver_instances_test.py tests/cwrmf_solver_replay_test.py`:
  **59 passed in 3.26 s**; collection counts 41 / 5 / 13 as reported, and
  `HEAD:paper_5/tests/test_apcm_carried_witness.py` has 33 `def test_`
  against 41 now, so the claimed eight new tests are there.
- full suite under `ram_guard --limit-mb 4000`: **428 passed, 9 warnings in
  185.58 s**, peak `1635 MB` (reported `428 passed ... 187.08 s`, peak
  `1972 MB`).

## Evidence re-derived from the artifacts

Everything below I computed myself from the named artifacts, not from the
report.

**(a) the per-atom table.**  From
`output/local_runs/paper_v_cwrmf_record_strict_t001_h0025_20260915_v1/conic_instances/instances.json`,
36 recorded instances, "rank" the positive-semidefinite cone side dimension
halved and seconds Clarabel's own `solve_time`: atom 0 critical Schur, 3
iterations at rank 6, `0.018 s`; atom 1 ranks 6,8,13,17,18,19,20,21,22,23,24,
25,26, `11.240 s` total of which `9.252 s` before the accepting solve; atom 2
26->31, 6 solves, `19.509 s` / `15.422 s`; atom 3 31->35, 5, `26.869` /
`20.298`; atom 4 35->36, 2, `15.614` / `5.713`; atom 5 36->38, 3, `27.300` /
`16.033`; atom 6 38->39, 2, `19.921` / `8.622`; atom 7 39->40, 2, `25.346` /
`10.884`.  Every cell of the report's table reproduces.  Cone side dimension
runs 12 to 80, interior-point iterations 12 to 28, `solve_time` `0.00595 s` to
`14.4618 s`.

**the negative result, re-run.**  I re-ran the implementer's own harness
`output/local_runs/paper_v_bundle_rank_seed_20260915_v1/seeded_probe.py` for
atom 1 at seeded ranks 26 and 40 under `ram_guard --limit-mb 4000`
(peak `1407 MB`), after reading it against
`_solve_relaxed_readable_bundle_cone` and confirming it rebuilds the same
compressed problem and evaluates the same full-Gram acceptance test
(`eigenvalues[0] >= schur_safety_margin - 1e-13` and `normalized_residual <=
10 * solver_tolerance`, with `schur_safety_margin = 2e-12` and
`solver_tolerance = 1e-10` from the run's own `numerical_settings`):

| seeded rank | status | s | accepted | full-Gram minimum | normalized residual |
|---:|---|---:|---|---:|---:|
| 26 | Solved | 1.833 | no | `-6.5304e-09` | `6.5324e-09` |
| 40 | Solved | 12.917 | yes | `+8.6723e-12` | `1.0446e-13` |

The round's load-bearing negative result stands as stated: a working set of
the shifted 62-row predictor Gram's own 26 lowest eigenvectors -- the rank the
recorded walk accepted at -- does not pass the strict acceptance test at that
atom.

**the compressed-dual refutation.**  Over the 26 discarded bundle solves of
the recorded rollout: statuses `Solved` and `AlmostSolved` only, maximum
compressed primal residual `4.9156e-11` and maximum compressed dual residual
`8.0999e-14`, both inside the acceptance bound `10 * solver_tolerance = 1e-9`.
Instance 22 (atom 3, first solve, rank 31) is `Solved` with `r_prim
3.1043e-15` and `r_dual 6.8529e-14` as reported.  The anchor's
`gram_eigendecomposition` tier is `0.0470268 s` of its `139.883 s` autonomous
wall time, so "`0.047 s` of `139.883 s`" is exact.

**the option table.**  Every row's calls, tier seconds, wall seconds,
all-coordinate scalar RMS, maximum readable-rate residual and maximum
completion correction norm reproduce from each run's `summary.json`, and I
recomputed the archive and carried columns myself as
`max |approximate_archive_coordinates_run - anchor|` (shape 5 x 31) and
`max |carried_states_run - anchor|` (shape 5 x 384).  Spot values:
`carried_accepted_basis` margin 2 `27` calls / `104.227 s` / archive
`1.4433e-15` / carried `1.0845e-06`; margin 2 + `max_threads_8` `28` /
`74.899 s` / `1.7417e-15` / `7.9475e-07` / readable `4.2952e-07`; margin 0 +
`max_threads_8` `22` / `44.872 s` / readable `1.1781e-05`, above the declared
`readable_rate_tolerance = 1e-5`, which the report flags.  The
`bundle_growth_factor 1.5` trigger-`0.0` run indeed has `success=false` and
`last_time=0.005` of `0.01`.  Speedup ratios recomputed against the stated
brackets agree to the reported two digits.

**the carried-working-set per-atom table.**  From
`output/local_runs/paper_v_cwrmf_seed_carriedbasis_record_strict_t001_h0025_20260915_v1/conic_instances/`:
22 instances, `98.395 s`; atom 1 13 solves 6->26 `11.556 s`, then atoms 2--7
one solve each at ranks 30, 34, 38, 42, 45, 45 costing `3.420`, `5.106`,
`11.176`, `25.030`, `21.347`, `20.740 s`.  That recorded run reproduces the
non-recording `carried_accepted_basis` run in every array except
`maximum_atom_seconds`.  The margin-2 column reproduces from that run's
`progress.jsonl`: accepting iterations 13, 4, 3, 1, 1, 1, 1 at ranks 26, 31,
35, 37, 39, 41, 43, i.e. 24 bundle solves plus 3 critical-Schur calls = the 27
`clarabel_solve` calls its summary reports.

**the balanced rollout.**  Parsing
`output/local_runs/paper_v_cwrmf_balanced_default_t005_h0025_20260915_v1/progress.jsonl`,
the balanced default's progress log reports 21 bundle atoms; the accepting
iteration histogram is 16 at iteration 1, 4 at iteration 2, 1 at iteration 6,
so 30 bundle solves of which 9 are discarded -- exactly as reported.  The
carried margin-2 balanced run's per-step active critical modes and `atom_s`
reproduce cell for cell (33/8.972, 35/4.218, 41/9.732, 49/32.087, 55/72.852,
39/9.238, 47/26.503, 53/66.454, 34/3.806, 38/6.960, 45/20.440).  Between the
two balanced runs the maximum absolute difference of the 31 reported archive
coordinates is `4.3093e-10` and of the 384 carried coordinates `1.1243e-03`,
as reported; `879.775 / 257.439 = 3.417`, the reported `3.42x`.

**impact analysis.**  Re-run read-only from the main checkout's index:
`_solve_relaxed_readable_bundle_cone` returns `risk LOW`, `epistemic exact`,
`impactedCount 3`, upstream `_solve_relaxed_readable` -> `radial_atom` ->
`integrate_cwrmf_ssprk2`, matching the recorded table.  No `UNKNOWN` verdict
to resolve.

## Corrections

1. **`86.242 s` is the wrong total for the 26 discarded solves.**  The 26
   discarded adaptive bundle-tier solves sum to `86.224 s` of Clarabel
   `solve_time`; `86.242 s` is that plus the three critical-Schur iterations'
   `0.018 s`, which are not among the 26.  The per-atom table's total row and
   the structured report's first result both carry the mixed figure.  The
   difference is `0.018 s`, 0.02 percent of the quantity, and changes nothing;
   the accepting column's `59.575 s` is `59.576 s` exactly (a sum of rounded
   cells).
2. **The `--bundle-seed-rank-margin` help text contradicts the
   recommendation.**  In `apcm_carried_witness_analysis.py` the flag's help
   reads "Declared margin added to the previous accepted bundle rank by the
   previous_accepted_rank_margin seed policy.  Unused by every other policy."
   `_bundle_seed_working_set` also reads `bundle_seed_rank_margin` in the
   `carried_accepted_basis` and `carried_accepted_basis_exit` branch, as the
   cap that trims the carried working set to the previous accepted rank plus
   the margin -- and the round's single recommendation is
   `--bundle-seed-policy carried_accepted_basis --bundle-seed-rank-margin 2`.
   The report's prose is right and the code is right; the flag's own help
   would tell a later reader that margin 2 is inert under that policy.
3. **Margin 4 and margin 0 are the same run, and the report does not say
   why.**  The table lists `carried_accepted_basis` margin 4 and margin 0 with
   identical RMS, archive, carried, readable and correction columns and 22
   calls each.  Verified reason: the trim branch is entered only when
   `bundle_seed_rank_margin > 0` and the working set exceeds the cap, and the
   untrimmed carried working set grows over the previous accepted rank by 4,
   4, 4, 4, 3, 0 along this rollout (26->30->34->38->42->45->45), so a cap at
   previous rank plus 4 never binds.  One sentence would spare the next reader
   the check.
4. **Two different default brackets are used as the basis in two places.**
   The option table's speedups are against `152.448 s` (bracket A); the
   "Dropped" paragraph compares `spectral_threshold` against `152.563 s`
   (bracket B).  Both brackets are declared and the ratio is unchanged at two
   digits, but the basis should be one of them throughout.

## Vocabulary

No invented quantity name found.  The report uses the record's own terms
throughout -- radial atom, working set, rank hint, accepting rank, discarded
solve, full 62-row Gram, shifted predictor Gram, completion velocity,
readable-rate residual, completion correction norm, critical mode limit,
`schur_safety_margin`, `psd_inflation`, `solver_tolerance` -- and every number
I checked carries its definition, its configuration and its horizon in the
same sentence or the same table caption.  The one place a repository field is
renamed in prose is the balanced table's "rank" column, which is the run's
`critical=` progress field; that field is `maximum_bundle`, the maximum
working-set rank reached within the atom, so the name is accurate.
