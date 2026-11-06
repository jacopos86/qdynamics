# review-outer-selector-formulation (2026-09-15)

Independent review of the round-3 implementer key `outer-selector-formulation`.
Verdict: **pass**.  Every gate holds, and the headline was reproduced from the
unmodified recorded routine without using the implementer's module at all.

## What was verified, and how

### 1. Scope

`git status --short` in the worktree lists twelve modified files and six
untracked ones.  Every modified file belongs to another round-3 key, by that
key's own ownership list, and every one is inside section 7's writable paths:

| modified file | owner this round |
|---|---|
| `paper_5/src/paper5/stability/apcm_carried_witness.py` | bundle-rank-seed |
| `paper_5/src/paper5/stability/apcm_carried_witness_analysis.py` | bundle-rank-seed |
| `paper_5/tests/test_apcm_carried_witness.py` | bundle-rank-seed |
| `paper_5/src/paper5/stability/projected_apcm.py` | projective-face |
| `paper_5/src/paper5/stability/apcm_moment_projection.py` | projective-face |
| `paper_5/src/paper5/stability/apcm_projective_event_guard_analysis.py` | projective-face |
| `paper_5/tests/test_projected_apcm.py` | projective-face |
| `paper_5/tests/test_apcm_moment_projection.py` | projective-face |
| `paper_5/tests/test_apcm_projective_event_guard.py` | projective-face |
| `paper_5/src/paper5/stability/cone_long_horizon.py` | joint-gram-t20 |
| `pipelines/open_dynamics/run_paper_v_energy_neutral_barrier_long_horizon.py` | joint-gram-t20 |
| `paper_5/tests/test_cone_long_horizon.py` | joint-gram-t20 |

This key's own additions are exactly the two files it was given
(`paper_5/src/paper5/stability/apcm_outer_selector_formulations.py`,
`paper_5/tests/test_apcm_outer_selector_formulations.py`), its report, and
eight `output/local_runs/paper_v_outer_selector_*_20260915_v1/` directories.
`apcm_projective_guard.py` carries md5 `03efd75c341235afa764d9ee5c07dfd1`,
which is the digest every one of the implementer's 47 measurements recorded,
and it is not in the modified list.  No scope violation.

### 2. Default path unchanged

No existing file was edited, and `grep -rln apcm_outer_selector_formulations`
over the repository returns only the new module and its own test file: nothing
on any executable path imports it.  The module makes no assignment into an
imported module and defines no module-level state beyond two type aliases and
`__all__`.  `OuterTargetFormulation()`'s ten fields are the recorded program
and are pinned field by field by
`test_default_formulation_states_the_recorded_program`.  Its copy of the
recorded Clarabel settings was read line by line against
`_solve_explicit_target_qp`'s own settings block
(`apcm_projective_guard.py:1327-1352`) and is identical, including the
profile-2 literals (conic tolerance `1e-10`, `max_iter` 500, equilibration
off, 20 iterative refinement passes at `1e-14`).

### 3. Tests rerun

| gate | implementer reported | this review |
|---|---|---|
| `paper_5/tests/test_apcm_outer_selector_formulations.py` | 14 passed in 1.75 s | 14 passed in 1.71 s |
| whole `paper_5` suite under `ram_guard --limit-mb 5000` | 425 passed, 9 warnings, 188.23 s, peak 1413 MB | 428 passed, 9 warnings, 189.15 s, peak 1655 MB |

The three extra tests are other round-3 keys' additions landing after the
implementer's gate run; nothing failed.

### 4. Parity evidence, re-derived independently

The claim under test is that the recorded one-shell outer target program at
`_solve_explicit_target_qp`'s own tightened retry profile returns a witness
the unchanged `1e-6` exploratory acceptance admits at the cutoff-16 correlated
preparation.  This review rebuilt the preparation's operands from the existing
builders only (`SymmetryReducedPositiveExtension`,
`exact_holstein_joint_moment_initial_state` at phonon cutoff 16,
`state_lower_moments`, `compile_invariant_target_readout`,
`projective_guard_outer_extension`, `_outer_current_image_matrix`, the inverse
readout covariance) and called the unmodified
`apcm_projective_guard._solve_explicit_target_qp` twice: once as the selector
calls it today, once at `retry_profile=2`.  15 entrance keys,
`lambda_ep = 1.5`, `gamma = 0.5`, drive amplitude 1, `t = 0`, boxed, frontier
weight 0, one evaluation, 60 one-shell outer Gram rows, 238 outer frontier
moments, under `ram_guard --limit-mb 5000` (peak 3634 MB).

| quantity | recorded ladder, as the selector calls it | `retry_profile=2` |
|---|---:|---:|
| status | `explicit_target:NumericalError` | `explicit_target:AlmostSolved` |
| acceptance under the unchanged `1e-6` rule | rejected | `provisional_independent_kkt` |
| objective, the invariant target image quadratic | `0.011286373848805029` | `0.011267123150739314` |
| minimum scaled eigenvalue of the reconstructed outer Gram | `-2.5903862151141617e-09` | `-2.1330616155590175e-14` |
| independent feasibility / stationarity / complementarity | `2.0375e-08` / `6.411437065053453e-05` / `1.0880e-07` | `3.4077e-10` / `3.863095431846568e-08` / `9.2285e-09` |
| relative duality gap | `3.5739542823707714e-04` | `7.46336974544425e-07` |
| wall seconds | `48.95` | `75.80` |

The first column reproduces the implementer's recorded row, and rounds 1 and
2, to every printed digit.  The second confirms the headline: the acceptance
is the recorded routine's own, with no tolerance argument passed, and it holds
with the image error counted (the recorded routine's feasibility includes it:
`3.408e-10`).

The 238-coordinate frontier point my run returned is **bitwise equal**
(`np.array_equal`, maximum absolute difference `0.0`) to the point the
implementer saved as
`output/local_runs/paper_v_outer_selector_formulations_prep_20260915_v1/point_recorded_kernel_retry2_w0.npz`.
Their module's assembly of the recorded program at profile 2 is therefore the
recorded program, not a lookalike.

Restriction of that point to the current cone, recomputed here from the saved
point and the rebuilt extensions: minimum scaled eigenvalue of the current
24-row Gram `-7.466675726570977e-16`, invariant target image maximum departure
`0.0`, outer envelope ratio `0.999463334347004`, current envelope ratio
`0.9927388271978883` — the report's four numbers, to every digit it printed.
The `envelope_fallback` threshold the report compares them with is
`1 - 10 * conic_tolerance = 1 - 1e-8` (`projected_apcm.py:1317-1318` with
`conic_tolerance = 1e-9`), as stated.

Conditioning claims, recomputed on the rebuilt operands: target metric
condition number `1613.8823529411766`; embedded readout singular values
`165.6381598545456` down to `4.123105625617661`; `R^T M R` has exactly 15
eigenvalues within `1e-9` of one and 223 within `1e-9` of zero out of 238.
All as reported.

`formulations.jsonl` in the three run directories was read row by row against
the report's three tables: every status, acceptance, objective, minimum scaled
eigenvalue, independent residual, relative duality gap and wall time in the
report matches its row.  The operand sets are saved as the task asked
(`scaled_readout`, `target_metric`, `real_base`, the 238 coefficient matrices,
`base_vector`, lower values, frontier and word scales, plus `operands.json`
naming each).

### 5. Comparability under other agents' edits

`projected_apcm.py` enters set B's operand build only.  It carried three
digests during the implementer's window and has changed once more since their
end-of-round recheck (now `36e1d6ba434bd69b120db5a8b83b3945`).  Set B's
operands were rebuilt here under the current tree: the 29 lower values and the
embedded readout are **bitwise equal** to the saved
`output/local_runs/paper_v_outer_selector_operands_stage_20260915_v1/operands.npz`.
Set B's rows remain comparable.

### 6. Vocabulary

Every quantity reported carries its definition, configuration and horizon, and
every name is the record's: relative duality gap, minimum scaled eigenvalue,
the invariant target image quadratic, independent KKT residuals, retry
profile, frontier weight, `psd_inflation`, lower moments, one-shell outer
cone, `provisional_independent_kkt`, `envelope_fallback`.  `outer envelope
ratio` is the repo's `selector_envelope_ratio` computed on the outer frontier
instead of the current one, and the harness states that computation
explicitly, so it is a stated quantity rather than an invented name.

## Two things the next round needs, neither of them a defect here

1. **Reaching the tightened profile inside the recorded routine is not one
   line.**  The report's attribution to the retry trigger set
   (`retry_statuses = {"AlmostSolved", "InsufficientProgress",
   "MaxIterations"}`, `apcm_projective_guard.py:1493`) is correct and is
   pinned by a test.  But the recorded routine has a second branch: at
   `retry_profile == 2` with status `AlmostSolved`, boxed, frontier weight 0
   and `accept_first_provisional` false, it hands off to
   `_solve_explicit_target_on_relative_face` and returns that result whenever
   it is not `None` (`apcm_projective_guard.py:1512-1532`).  This review had
   to pass `accept_first_provisional=True` to isolate the profile-2 point.  So
   adding `NumericalError` to the trigger set alone would not necessarily
   return the accepted witness measured here; the implementer's callable is
   unaffected because it never calls the recorded routine.  The report does
   not mention this branch.
2. **The module scores candidates with the image equality satisfied
   exactly.**  `evaluate_on_recorded_program` sets `q := R z`, so the image
   error is zero by construction, where the recorded routine carries `q` as a
   solver variable and counts its residual.  That is why the module's set-A
   recorded row reads objective `0.011286373843433770` and feasibility
   `2.5904e-09` where the recorded routine reads `0.011286373848805029` and
   `2.0375e-08`.  The report prints both.  The substitution did not
   manufacture the acceptance: the recorded routine's own diagnostics accept
   the same point (feasibility `3.408e-10`, gap `7.46336974544425e-07`).

One reproducibility gap, minor: the restriction numbers (the accepted
witness's restriction to the current cone and the two envelope ratios) live in
the report and in `restrict_accepted_witness.py` but in no saved output file.
They were recomputed here and hold to every printed digit.

## Gate summary

| gate | result |
|---|---|
| scope: only owned files changed, all inside section 7 | pass |
| defaults untouched; new behaviour only behind named options | pass |
| owned test file rerun | pass, 14 passed |
| whole `paper_5` suite rerun under the memory guard | pass, 428 passed, peak 1655 MB |
| parity evidence re-derived by the reviewer | pass, frontier point bitwise equal; recorded first pass reproduced digit for digit |
| headline reproduced without the new module | pass, `provisional_independent_kkt` at gap `7.46336974544425e-07` under the unchanged `1e-6` rule |
| every number carries definition and horizon; no invented quantity name | pass |
