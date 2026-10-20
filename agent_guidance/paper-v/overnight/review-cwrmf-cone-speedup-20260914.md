# Independent review of `cwrmf-cone-speedup`, 2026-09-14

Reviewer agent.  No source file edited.  Worktree
`/Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914`,
branch `claude/paper-v-routes-20260914`.

## Verdict

Pass.  Every gate the task set holds under independent re-execution: scope,
the two cited test commands, bitwise parity on both anchors for all six
staged changes, defaults untouched, and every reported number matched to its
`summary.json` with its definition and horizon.  Four non-blocking findings
are recorded at the end; none of them changes a number or a default.

## Scope

`git status --short` shows seven modified tracked files.  Four are the
implementer's, and all lie inside the section 7 allowlist:

- `paper_5/src/paper5/stability/apcm_carried_witness.py`
- `paper_5/src/paper5/stability/apcm_carried_witness_analysis.py`
- `paper_5/src/paper5/stability/apcm_positive_extension.py` (only
  `_clarabel_svec_upper` and the new `_clarabel_svec_upper_pattern`)
- `paper_5/tests/test_apcm_carried_witness.py`

The other three (`apcm_projective_guard.py`, `cone_correction.py`,
`projected_apcm.py`) and the two untracked `*_projective_event_guard*` files
belong to the concurrent `projective-event-guard` agent, whose own report
claims them; a diff of those three for this task's fingerprints
(`tier_seconds`, `batched_mode_gradients`, `_step_scaled_completion`,
`svec_upper`, `maximum_step_subdivisions`, `commutator_row`) returns nothing.
The report file under `agent_guidance/paper-v/overnight/` and every run
directory under `output/local_runs/paper_v_cwrmf_*` are allowlisted.  The main
checkout is untouched: `/Users/lordskyler/Holstein_test/paper_5/...` still
carries its 2026-08-31 20:31 mtimes.  No scope violation.

## Tests, rerun verbatim

| command | result |
|---|---|
| `cd paper_5 && PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/test_apcm_carried_witness.py` | 30 passed in 2.40 s |
| `cd paper_5 && PYTHONPATH=src python -m pytest -q -p no:cacheprovider` | 319 passed, 9 warnings in 109.37 s |

Both match the reported counts.  The diff adds exactly eight test functions,
consistent with 22 -> 30 and 311 -> 319.

## Parity, re-checked by loading the arrays

Loaded every array of `autonomous_trajectory.npz` and `trajectory.npz` myself
and compared the pre-change reference against each staged run with
`np.array_equal(..., equal_nan=True)`, excluding only `maximum_atom_seconds`
(wall time).

| anchor | instr | b4 | b1 | b2 | c | final |
|---|---|---|---|---|---|---|
| no guard, t = 0 to 0.05, h = 0.0025 (20 steps, 40 atoms) | bitwise | bitwise | bitwise | bitwise | bitwise | bitwise |
| strict, t = 0 to 0.01, h = 0.0025 (4 steps, 8 atoms) | bitwise | bitwise | bitwise | bitwise | bitwise | bitwise |

`velocity_margins` in the no-guard pair carries 20 NaN entries in both runs;
its NaN positions coincide and its uint64 bit patterns are equal, so it is
bitwise identical and not an exception.  No stated tolerance was needed
anywhere: the worst difference on the guarded path is exactly zero, so the
Clarabel-determinism fallback the task allowed was never invoked.  Reference
directories are dated 22:39 and 22:42, ahead of every staged run
(22:47 through 23:12), consistent with references saved first.

## Numbers re-read from `summary.json`

Every value in the implementer's results list matches the artifact.  Spot
checks, each over t = 0 to 0.01 at h = 0.0025 in the strict profile
(4 SSPRK2 steps, 8 radial atoms, lambda = 1.5, gamma = 0.5, drive amplitude
1, phonon cutoff 16, exact-correlated preparation) unless stated:

- Clarabel solve 142.35022 s of a 145.17260 s rollout over 36 solver calls,
  which is 98.06%; coefficient assembly and svec packing 2.34537 s (1.62%);
  62-row eigendecomposition 0.04860 s; unguarded velocity evaluation
  0.14874 s; retraction 0 calls.
- Coefficient assembly and svec tier 2.34537 s -> 0.57062 s after the
  vectorized packer.
- Strict autonomous wall 143.51106 s -> 139.88282 s (1.026x).
- No-guard autonomous wall 1.28636 s -> 0.63140 s (2.04x) and rollout
  0.92839 s -> 0.29642 s, over t = 0 to 0.05 at h = 0.0025; the unguarded
  velocity tier 0.73860 s -> 0.11393 s over 40 radial atoms (6.5x).
- All-coordinate scalar RMS against the regenerated exact cutoff-16
  reference: 5.836875e-05 no guard and 5.827262e-05 strict, identical in the
  reference and in every staged run.

Independent re-measurement of the packer in a fresh process: bitwise equal to
the literal double loop at sizes 1, 2, 3, 11, 62, 80, 124, 127 on symmetric,
nonsymmetric and 1e-18/1e18-scaled inputs; 340 packs of one 80x80 matrix took
0.1007 s literal against 0.0034 s vectorized (29.3x here, 31.5x reported).

## Defaults

- `CWRMFSettings.batched_mode_gradients` is `False`
  (`apcm_carried_witness.py:139`), and `cwrmf_settings_for_profile` sets it in
  neither the strict nor the balanced branch (`201-230`).
- `integrate_cwrmf_ssprk2(..., maximum_step_subdivisions=0)`; at zero the
  subdivision loop is not entered and the three failure message strings, the
  atom-evaluation accounting and the progress line are character-for-character
  the previous ones (read line by line against the removed block).
- Tier timing is `None` until `enable_tier_timing()` is called; the driver
  calls it only under `--tier-timing`, after `prepare`.
- The three new driver flags all default to the recorded behaviour.
- The strongest evidence is the parity table above: the `final` runs were
  produced with no flags set and reproduce the pre-change arrays bitwise.

## Impact analysis (run by the reviewer, read-only, against the main index)

The implementer's report records no impact analysis; run here for the edited
symbols, `node /Users/lordskyler/Holstein_test/.gitnexus/run.cjs impact
"<name>" --direction upstream --repo /Users/lordskyler/Holstein_test` (the
index is 7 commits behind this worktree, so it is used as a caller hint and
every UNKNOWN is resolved by grep):

| symbol | risk | direct callers of note |
|---|---|---|
| `_clarabel_svec_upper` | MEDIUM, 22 impacted | `apcm_projective_guard.py` (`_solve_guard_qp`, `_solve_explicit_target_qp`, `_solve_guard_qp_on_relative_face`, `_solve_explicit_target_on_relative_face`, `solve`), `apcm_moment_projection.py:_stage_cone_data`, `apcm_positive_extension.py:_completion_cone_data`, and the three CWRMF solver tiers |
| `_moment_derivative` | LOW, 8 | `retained_velocity`, `desired_completion_velocity` |
| `negative_mode_coupling` | LOW, 5 | `_event_guard_fast_path`, `apcm_carried_witness_mode_audit.py:main` |
| `integrate_cwrmf_ssprk2` | LOW, 4 | `apcm_carried_witness_analysis.py:main` |
| `_event_completion_retraction` | LOW, 3 | `_event_guard_fast_path` |
| `center_cross_value` | UNKNOWN, 0 | resolved by grep: no production caller outside its own module; the only remaining caller is the new test |

## Non-blocking findings

1. **No impact analysis was recorded before editing.** The protocol asks for
   it in the report; the table above supplies it after the fact.  The one
   regression the work actually hit -- `negative_mode_coupling` reading
   `model.settings` on the duck-typed `_CouplingModelAdapter`
   (`projected_apcm.py:682`) -- is exactly the plain-object caller the
   UNKNOWN/MEDIUM verdicts warn about.  It was caught by the full suite and
   fixed with `getattr`, and the fix is in the implementer's own file.
2. **`_clarabel_svec_upper` is shared with the concurrent agent's files.**
   Four call sites in `apcm_projective_guard.py` and one in
   `apcm_moment_projection.py` now run the vectorized packer.  Values are
   bitwise unchanged (independently verified above), but any URPG wall-time
   line measured before 22:55 today and compared with one measured after is
   comparing two different packers.  Worth one sentence to the
   `projective-event-guard` agent.
3. **`test_center_cross_evaluator_matches_the_one_shot_reader` is
   tautological.** `center_cross_value` now delegates to
   `center_cross_evaluator`, so the test compares the evaluator with itself.
   Its consequence is small -- grep shows the wrapper has no production
   caller, and the real gate for that change is the bitwise no-guard rollout,
   which passes -- but the report lists it as a gate and it is not one.
4. **Two numbers have no saved artifact.** The svec microbenchmark (31.5x)
   and the per-mode/batched timing table in change 5 cite the test file,
   which checks values and not time.  Both are cheap to reproduce; the svec
   one is reproduced above at 29.3x.

## Not in evidence

`_solve_relaxed_readable_full_cone` and `_polish_terminal_completion_velocity`
were rewired to the cached step-scaled tensor but are exercised by neither
anchor (the strict profile leaves `enable_full_cone_fallback` off) nor by any
test.  The cached array is marked read-only; a grep for `coefficients[...] =`
finds writes only inside `_compile_scaled_completion_coefficients`, the
builder, so no consumer mutates it, but that is an argument rather than a run.
The balanced event-guard rollout of plan step 5 would exercise both.
