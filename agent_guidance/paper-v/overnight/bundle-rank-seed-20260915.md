# Bundle rank seed (key `bundle-rank-seed`), 2026-09-15

Worktree `/Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914`,
branch `claude/paper-v-routes-20260914`, HEAD `d189c50a`, round 3 of the
overnight loop.  Round-2 finding this attacks: 26 of the 33 adaptive
bundle-tier Clarabel calls over the strict `t=0` to `0.01` rollout at
`h=0.0025` are enlargement iterations whose solution is discarded, and
`_bundle_rank_hint` starts at 0, so the first bundle atom walked the
positive-semidefinite cone side dimension from 12 to 52 in `11.24 s` before
the atom that used it started at 52.  Nothing is committed; strict and
balanced defaults are unchanged.

## Negative result first

The round-2 reading that the enlargement iterations are discarded work does
not survive measurement.  The working set the enlargement builds is not the
predictor's own leading eigenvector set at the same rank: seeding the first
bundle atom's working set with the 26 lowest eigenvectors of its shifted
62-row predictor Gram -- the rank at which the recorded walk accepted --
gives a full-Gram minimum eigenvalue of `-6.5304e-09` against the declared
acceptance floor `schur_safety_margin - 1e-13 = 1.9e-12`, so it does not
accept; the lowest seeded rank that accepts at that atom is 40, at
`12.841 s` against the recorded walk's whole `11.240 s`, and seeded ranks
46 and 52 return `NumericalError`.  The enlargement iterations are how the
working set finds the directions in which the correction drives the 62-row
Gram negative, and those directions are not in the predictor's own low
spectrum.

What is recoverable is the *continuation* across radial atoms.  Carrying the
previous atom's accepted working set instead of only its rank removes every
discarded solve after the first bundle atom -- atoms 2 through 7 then accept
on their first iteration -- and the strict `t=0` to `0.01` rollout falls from
`152.4` -- `152.8 s` of Clarabel tier to `98.428 s`.  Trimming that carried
working set back to the previous accepted rank plus 2, which keeps the
reported feasibility quantities in the anchor's band, gives `104.227 s`
(`1.46x`) alone and `74.899 s` (`2.04x`) with round 2's named
`max_threads_8` profile on top; untrimmed and with threads it is `44.872 s`
(`3.40x`), at a maximum readable-rate residual of `1.1781e-05` against the
declared `readable_rate_tolerance = 1e-5`.  In every case the 31 reported
archive coordinates agree with the anchor to `9.535e-14` or better.  All of it is off by default and the
default path reproduces the round-1 anchor bitwise.

## (a) How the adaptive bundle tier chooses and enlarges its working set

Code: `CarriedWitnessModel._solve_relaxed_readable_bundle_cone`, at
`paper_5/src/paper5/stability/apcm_carried_witness.py:2249` before this
round's edits and `:2424-2737` after them; the seed is now
`_bundle_seed_working_set` at `:1066` and the growth target
`_bundle_growth_target` at `:184`.

1. **The Gram it works on.** `shifted_predictor` is the scaled unified 62-row
   Gram of the unconstrained endpoint `(endpoint_retained, completion +
   step_size * desired)` plus `psd_inflation` on the diagonal (`1e-11`
   strict).  It is fixed for the whole tier call.
2. **The rank hint.** `spectral_rank` is the number of shifted predictor
   eigenvalues below `spectral_entry_threshold` (`1e-8` strict).  The
   starting rank is
   `required_rank = min(critical_mode_limit, max(spectral_rank,
   self._bundle_rank_hint))` and the starting working set is
   `predictor_vectors[:, :required_rank]`, the lowest `required_rank`
   eigenvectors.  `_bundle_rank_hint` is initialised to 0 and updated on
   acceptance by `self._bundle_rank_hint = max(self._bundle_rank_hint,
   maximum_bundle)`, so it is monotone along the rollout and the working set
   never shrinks.
3. **The compressed problem.** With `B` the working set,
   `compressed_base = B^H G B - schur_safety_margin I` and
   `compressed_coefficients[j] = B^H C_j B` for each of the 340 step-scaled
   completion coefficient matrices.  Both are realified (`_realify_hermitian`
   doubles a complex Hermitian block) and scaled by `1e4`, so the
   positive-semidefinite cone side dimension is `2 * rank`.  The second-order
   cone carries the velocity ceiling.  The objective is
   `0.5 x^T diag(w) x + 1e-6 desired^T x` with `w = 1e-6` off the readable
   completion indices and `1 + 1e-6` on them.
4. **The acceptance test.** It is on the *full* 62-row Gram, not the
   compressed one: `candidate_matrix = shifted_predictor + sum_j x_j C_j`,
   `eigenvalues = eigh(candidate_matrix)`, and the solution is accepted when
   `eigenvalues[0] >= schur_safety_margin - 1e-13` and
   `normalized_residual <= 10 * solver_tolerance`, where
   `normalized_residual = max(0, schur_safety_margin - eigenvalues[0],
   radial_residual, r_prim, r_dual) / (1 + |eigenvalues[0]|)` and `r_prim`,
   `r_dual` are the compressed problem's own primal and dual residuals.
5. **The enlargement rule.** On rejection the working set is augmented with
   the eigenvectors of the *full candidate* matrix whose eigenvalue is below
   `schur_safety_margin`, the augmented block is orthonormalised by an SVD,
   and the new rank is the number of singular values above `1e-10`.  If that
   rank does not exceed the current one the tier stops enlarging and hands
   the candidate to `_polish_terminal_completion_velocity`.  A failed or
   non-`Solved`/`AlmostSolved` return backs the rank off once to
   `max(spectral_rank, rank // 2)`.

### The 36 recorded instances, per radial atom

Recorded rollout: strict numerical profile, always-on guard, `t=0` to `0.01`
at `h=0.0025` (4 SSPRK2 steps, 8 radial atoms), strong coupling
`lambda = 1.5`, `gamma = 0.5`, drive amplitude 1, phonon cutoff 16,
exact-correlated preparation.  Artifact
`output/local_runs/paper_v_cwrmf_record_strict_t001_h0025_20260915_v1/conic_instances/`.
Rank is the positive-semidefinite cone side dimension halved; seconds are
Clarabel's own `solve_time`.  Atom 0 is the critical-Schur tier, whose three
SQP iterations did not converge and whose accepted velocity came from the
SLSQP refinement that follows them, so it has no bundle row.

| atom | time | starting rank | accepting rank | solves | discarded solves | discarded s | accepting s | total s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.00000 | -- (critical Schur, 3 iterations at rank 6) | -- | 3 | 3 | 0.018 | -- | 0.018 |
| 1 | 0.00250 | 6 | 26 | 13 | 12 | 9.252 | 1.988 | 11.240 |
| 2 | 0.00250 | 26 | 31 | 6 | 5 | 15.422 | 4.087 | 19.509 |
| 3 | 0.00500 | 31 | 35 | 5 | 4 | 20.298 | 6.571 | 26.869 |
| 4 | 0.00500 | 35 | 36 | 2 | 1 | 5.713 | 9.901 | 15.614 |
| 5 | 0.00750 | 36 | 38 | 3 | 2 | 16.033 | 11.267 | 27.300 |
| 6 | 0.00750 | 38 | 39 | 2 | 1 | 8.622 | 11.299 | 19.921 |
| 7 | 0.01000 | 39 | 40 | 2 | 1 | 10.884 | 14.462 | 25.346 |
| **total** | | | | **36** | **26** | **86.242** | **59.575** | **145.817** |

The accepting rank advances 26, 31, 35, 36, 38, 39, 40 and the starting rank
of each atom is the previous atom's accepting rank, because the hint is
monotone.  `86.242 s` of the `145.817 s` Clarabel reported for the rollout is
in solutions that were discarded; the seven accepting solves cost
`59.575 s`.  That difference is the whole prize, and it is an upper bound on
what any seeding or enlargement policy can return.

The rejected solves separate into two kinds.  Twenty-three miss the cone:
their full-Gram minimum runs from `-5.576e-08` (atom 1, rank 6) to
`-6.724e-11` (atom 7, rank 39).  Three are positive but short of the
declared interior margin -- instance 25 (atom 3, rank 34,
`+6.282e-13`), instance 27 (atom 4, rank 35, `+8.539e-13`) and instance 30
(atom 5, rank 37, `+1.586e-12`), all below `schur_safety_margin - 1e-13 =
1.9e-12` and all with `normalized_residual` at or below `4.916e-11`, so they
were rejected by the margin alone at a cost of `21.712 s`.

### What a seeded working set does instead, measured offline

Harness `output/local_runs/paper_v_bundle_rank_seed_20260915_v1/seeded_probe.py`
rebuilds one recorded atom's compressed problem with the working set taken as
the `rank` lowest eigenvectors of that atom's shifted 62-row predictor Gram --
which is exactly what the tier's first iteration carries for a seed of
`rank` -- solves it with the recorded Clarabel settings and evaluates the same
strict full-Gram acceptance test.  Seconds are the replay wall time on this
machine; `dv_max` is the maximum absolute difference of the returned
completion velocity from the one the recorded atom accepted.

| atom | recorded accepting rank | seeded rank | status | s | accepted | full-Gram minimum | dv_max |
|---:|---:|---:|---|---:|---|---:|---:|
| 1 | 26 | 20 | Solved | 0.555 | no | `-4.5408e-09` | `9.9399e-05` |
| 1 | 26 | 24 | Solved | 1.366 | no | `-2.8629e-09` | `1.0169e-04` |
| 1 | 26 | 26 | Solved | 1.658 | no | `-6.5304e-09` | `1.0665e-04` |
| 1 | 26 | 28 | Solved | 2.113 | no | `-2.1814e-09` | `6.1632e-05` |
| 1 | 26 | 31 | Solved | 3.000 | no | `-3.3935e-10` | `5.4868e-05` |
| 1 | 26 | 34 | Solved | 5.118 | no | `-3.6402e-11` | `1.0897e-04` |
| 1 | 26 | 40 | Solved | 12.841 | **yes** | `+8.6723e-12` | `9.2087e-05` |
| 1 | 26 | 46 | NumericalError | 17.142 | no | `-6.5763e-09` | `1.0872e-01` |
| 1 | 26 | 52 | NumericalError | 29.675 | no | `-4.8710e-11` | `9.1922e-01` |
| 2 | 31 | 28 -- 32 | Solved | 2.253 -- 3.777 | no | `-2.3597e-09` to `-3.3130e-10` | -- |
| 2 | 31 | 34 | Solved | 5.164 | no | `-1.6726e-11` | `1.1609e-04` |
| 2 | 31 | 36 | Solved | 7.071 | **yes** | `+6.6545e-12` | `6.8174e-05` |
| 2 | 31 | 38 | Solved | 9.207 | yes | `+4.5983e-12` | `1.7319e-04` |
| 2 | 31 | 40 | Solved | 13.449 | yes | `+8.4064e-12` | `1.2786e-04` |
| 7 | 40 | 40 | Solved | 13.587 | no | `-7.1639e-11` | `1.4353e-04` |
| 7 | 40 | 41 | Solved | 15.048 | no | `-1.3257e-11` | `2.3858e-04` |
| 7 | 40 | 42 | Solved | 14.597 | **yes** | `+8.7526e-12` | `7.9436e-05` |
| 7 | 40 | 44 | Solved | 22.205 | yes | `+2.8219e-12` | `2.3575e-04` |

Three readings.  A working set of the predictor's own lowest eigenvectors
needs 14 more directions than the walked one at atom 1 (40 against 26), 5 more
at atom 2 (36 against 31) and 2 more at atom 7 (42 against 40); the walked
working set is smaller because it contains the directions in which the
*correction* drives the Gram negative, which the predictor's low spectrum does
not carry.  At atom 1 the one seeded solve that accepts costs `12.841 s`
against the whole recorded walk's `11.240 s`, so seeding that atom is a loss;
at atoms 2 and 7 it is a gain (`7.071 s` against `19.509 s`, `14.597 s`
against `25.346 s`).  And the compressed problem itself degrades with rank:
at seeded ranks 46 and 52 Clarabel returns `NumericalError` and the returned
completion velocity is `1.1e-01` and `9.2e-01` away from the accepted one.

The number of endpoint unshifted 62-row Gram eigenvalues below a threshold is
nearly the same at every atom and does not track the accepting rank
(26, 31, 35, 36, 38, 39, 40):

| threshold | `1e-8` | `1e-7` | `1e-6` | `1e-5` | `1e-4` | `1e-3` | `1e-2` |
|---|---:|---:|---:|---:|---:|---:|---:|
| count, atom 1 | 6 | 10 | 23 | 33 | 37 | 41 | 45 |
| count, atoms 2--7 | 4 -- 6 | 8 -- 9 | 23 | 33 | 37 | 41 | 45 |

## (b) The named options, all off by default

Every option is a field on `CWRMFSettings` with a default that reproduces the
recorded rule, and a driver flag on
`paper5.stability.apcm_carried_witness_analysis`.  The gate on the default
path is at the end of this section: the strict `t=0` to `0.01` rollout with
all six fields at their defaults reproduces the round-1 anchor bitwise.

**(1) Seed policies** -- `bundle_seed_policy`, with
`bundle_seed_spectral_threshold` (default `1e-6`) and
`bundle_seed_rank_margin` (default 0).

| policy | starting working set |
|---|---|
| `monotone_hint` (default) | the recorded rule: the `max(spectral_rank, bundle_rank_hint)` lowest predictor eigenvectors |
| `spectral_threshold` | the same, seeded at the larger of the hint and the number of endpoint unshifted 62-row Gram eigenvalues below `bundle_seed_spectral_threshold` |
| `previous_accepted_rank` | seeded at the rank the previous bundle atom accepted at, without the monotone maximum |
| `previous_accepted_rank_margin` | that rank plus `bundle_seed_rank_margin` |
| `carried_accepted_basis` | the previous accepted working set itself, orthonormalised with the current predictor's leading `spectral_rank` eigenvectors; a positive `bundle_seed_rank_margin` trims the result to the `previous accepted rank + margin` lowest modes of its own compressed block |
| `carried_accepted_basis_exit` | the same, after dropping the carried modes whose unshifted compressed eigenvalue has risen to or above `bundle_seed_spectral_threshold` |

**(2) Enlargement policy** -- `bundle_growth_factor` (default `1.0`) and
`bundle_growth_trigger` (default `0.0`).  When a rejected solve's normalized
residual exceeds the trigger, the enlarged working set is padded with further
predictor eigenvectors until its rank reaches
`ceil(bundle_growth_factor * current rank)`, capped at the critical mode
limit.  At the default factor the padding branch is never entered.

**(3) The compressed problem's interior** -- `bundle_interior_overshoot`
(default `1.0`) multiplies the margin the *compressed* base is built with,
`B^H G B - schur_safety_margin * overshoot * I`.  The acceptance test is
untouched: `schur_safety_margin` and `solver_tolerance` keep their values, and
a repository test pins that.

### (c) part one: the strict `t=0` to `0.01` rollout at `h=0.0025`

Same configuration as the round-1 anchor throughout: strict numerical
profile, always-on guard, 4 SSPRK2 steps, 8 radial atoms, strong coupling
`lambda = 1.5`, `gamma = 0.5`, drive amplitude 1, phonon cutoff 16,
exact-correlated preparation, exact reference regenerated in process,
`--tier-timing` on.  "calls" and "tier s" are the `clarabel_solve` tier's
call count and seconds; "archive" and "carried" are the maximum absolute
differences of `approximate_archive_coordinates` (5 x 31) and
`carried_states` (5 x 384) from the anchor; "readable" is the maximum
readable-rate residual and "correction" the maximum completion correction
norm, whose anchor values are `1.4396e-07` and `2.9536e-06`.

The machine is shared with other overnight work, so the default path was
re-measured four times in this session as the comparison basis, spread
through it: `152.774 s`, `152.448 s`, `152.563 s` and `150.703 s` of Clarabel
tier against the round-1 anchor's `138.644 s`, a spread of `1.860 s` on about
`152 s`.  All four reproduce the anchor trajectory **bitwise** -- every array
of `autonomous_trajectory.npz` and `trajectory.npz` except
`maximum_atom_seconds` equal under `np.array_equal`.  The first was measured
before any edit of this round, the second after the seed, growth and
overshoot edits, the last two after every edit.  Speedups below are against
`152.448 s`.

| run | calls | tier s | wall s | speedup | all-coordinate scalar RMS | archive | carried | readable | correction |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| anchor, round 1 | 36 | 138.644 | 139.883 | -- | `5.827261740618364e-05` | -- | -- | `1.4396e-07` | `2.9536e-06` |
| default, this session (bracket A) | 36 | 152.448 | 153.910 | `1.00x` | `5.827261740618364e-05` | bitwise | bitwise | `1.4396e-07` | `2.9536e-06` |
| `previous_accepted_rank` | 36 | 151.318 | 152.729 | `1.01x` | `5.827261740618364e-05` | bitwise | bitwise | `1.4396e-07` | `2.9536e-06` |
| `previous_accepted_rank_margin`, margin 3 | 34 | 133.628 | 134.999 | `1.14x` | `5.827261740618363e-05` | `2.914e-16` | `4.464e-07` | `3.5233e-07` | `2.9727e-06` |
| `spectral_threshold`, `1e-4` | 14 | 130.913 | 132.033 | `1.16x` | `5.827261740618336e-05` | `9.914e-16` | `4.500e-07` | `5.1675e-07` | `2.9340e-06` |
| **`carried_accepted_basis`, margin 2** | **27** | **104.227** | **105.712** | **`1.46x`** | `5.827261740618362e-05` | `1.443e-15` | `1.084e-06` | `3.2478e-07` | `2.9541e-06` |
| `carried_accepted_basis`, no margin | 22 | 98.428 | 99.539 | `1.55x` | `5.827261740618419e-05` | `3.844e-15` | `2.871e-05` | `4.2544e-05` | `3.4271e-05` |
| `carried_accepted_basis`, margin 4 | 22 | 99.199 | 100.476 | `1.54x` | `5.827261740618419e-05` | `3.844e-15` | `2.871e-05` | `4.2544e-05` | `3.4271e-05` |
| `carried_accepted_basis_exit`, `1e-4` | 27 | 174.016 | 175.438 | `0.88x` | `5.827261740618418e-05` | `3.796e-15` | `1.436e-05` | `3.5629e-05` | `2.8353e-05` |
| `carried_accepted_basis_exit`, `1e-6` (default threshold) | 65 | 268.906 | 270.764 | `0.57x` | `5.827261740618362e-05` | `3.615e-15` | `2.069e-07` | `1.9619e-07` | `2.9251e-06` |
| `bundle_growth_factor` 1.5, trigger `1e-7` | 32 | 141.664 | 143.005 | `1.08x` | `5.827261740618336e-05` | `1.631e-15` | `7.149e-07` | `3.5009e-07` | `2.9201e-06` |
| `bundle_growth_factor` 1.5, trigger `0.0` | 20 | 333.470 | 334.806 | -- | stopped at `t=0.005` of `0.01` | -- | -- | `1.7050e-03` | `1.0738e-03` |
| `bundle_interior_overshoot` 3 | 38 | 138.078 | 139.514 | `1.10x` | `5.827261740618309e-05` | `1.249e-15` | `1.312e-07` | `1.5105e-07` | `2.9335e-06` |
| `bundle_interior_overshoot` 10 | -- | -- | -- | -- | first bundle atom returns `PrimalInfeasible` at rank 6; no step completed | -- | -- | -- | -- |

`previous_accepted_rank` reproduces the anchor **bitwise**, which is the
measured answer to whether the monotone hint and the previous accepting rank
differ on this rollout: they do not.  The hint is
`max(hint, maximum_bundle)` over the whole history, the accepted rank is the
last accepting one, and because the accepting rank rises monotonically here
and neither the rank back-off nor the terminal polish fires, the two
sequences coincide and every one of the 36 solves is the same problem.

### Why the working set, not its rank, is the lever

Per-atom bundle-tier table of the carried-working-set run, recorded with
`--record-conic-instances` (which reproduces the same trajectory):
artifact `output/local_runs/paper_v_cwrmf_seed_carriedbasis_record_strict_t001_h0025_20260915_v1/conic_instances/`.

| atom | recorded anchor: solves, ranks, s | `carried_accepted_basis`: solves, ranks, s | margin 2: solves, ranks, s |
|---:|---|---|---|
| 1 | 13, 6->...->26, 11.240 | 13, 6->...->26, 11.556 | 13, 6->...->26, 11.536 |
| 2 | 6, 26->...->31, 19.509 | 1, 30, 3.420 | 4, 28->29->30->31, 14.181 |
| 3 | 5, 31->...->35, 26.869 | 1, 34, 5.106 | 3, 33->34->35, 19.366 |
| 4 | 2, 35->36, 15.614 | 1, 38, 11.176 | 1, 37, 8.793 |
| 5 | 3, 36->37->38, 27.300 | 1, 42, 25.030 | 1, 39, 12.071 |
| 6 | 2, 38->39, 19.921 | 1, 45, 21.347 | 1, 41, 15.253 |
| 7 | 2, 39->40, 25.346 | 1, 45, 20.740 | 1, 43, 22.353 |
| total | 36 instances, 145.819 s | 22 instances, 98.395 s | 27 instances, 103.571 s |

Carrying the accepted working set removes **every** discarded solve after the
first bundle atom: atoms 2 through 7 accept on their first iteration.  The
first atom is unhelped because there is nothing to carry into it, and its 12
discarded solves cost `9.5 s` of the `98.4 s`.  What the carried working set
pays instead is rank: it accepts at 30, 34, 38, 42, 45, 45 where the anchor
accepted at 31, 35, 36, 38, 39, 40, and the solve cost rises steeply with
rank, which is why the gain is `1.55x` and not the `2.4x` the discarded
seconds alone would suggest.  Trimming the carried working set to the
`previous accepted rank + 2` lowest modes of its own compressed block
(`bundle_seed_rank_margin = 2`) holds the accepting rank to 31, 35, 37, 39,
41, 43, costs 5 more solves, and brings the maximum readable-rate residual
back to `3.2478e-07` and the maximum completion correction norm to
`2.9541e-06` against the anchor's `1.4396e-07` and `2.9536e-06`.  Untrimmed,
those two reported quantities are `4.2544e-05` and `3.4271e-05`, and the
first is above the declared `readable_rate_tolerance = 1e-5`; that tolerance
gates only the terminal polish tier's cone radius, which never fired here, so
nothing was violated, but the untrimmed working set is repairing along
directions the new endpoint does not need and the reported residual says so.
Margin 2 is therefore the qualified form of the option and margin 0 the fast
one.

### (3) The compressed problem's dual cannot certify the full cone

`clarabel` 0.11.1 does expose `solution.s` and `solution.z` alongside
`r_prim` and `r_dual`, so the code has the dual information.  It does not
help, for one structural and one measured reason.

Structural: the compressed problem's entire data is `B^H G B` and
`B^H C_j B`.  Two full 62-row Grams that agree on those blocks give
bit-identical compressed problems, primal and dual, and can differ freely on
the orthogonal complement of `span(B)`, so no function of the compressed
primal-dual pair separates them.  A certificate for the full cone needs the
complement, which is what the full eigendecomposition supplies.

Measured: all 26 discarded bundle solves of the recorded rollout return
`Solved` or `AlmostSolved` with compressed primal and dual residuals at or
below `4.916e-11`, inside the acceptance bound `10 * solver_tolerance = 1e-9`
-- while their full 62-row Gram minima run down to `-5.576e-08`.  The most
extreme is instance 22 (atom 3, first solve, rank 31): status `Solved`,
`r_prim = 3.104e-15`, `r_dual = 6.853e-14`, full-Gram minimum `-1.519e-08`.
A compressed-dual early acceptance would have accepted every solve the full
cone rejects.

And there is nothing to save even if such a certificate existed: the 62-row
eigendecomposition tier that does certify costs `0.047 s` of the anchor's
`139.883 s`, 0.03 percent.  The option is therefore not implemented; what is
implemented in its place is `bundle_interior_overshoot`, which attacks the
three near-miss solves from the other side by asking the compressed problem
for more interior without touching the acceptance test.  It does not pay:
at overshoot 3 the rollout takes 38 solves instead of 36, and at overshoot 10
the first bundle atom's rank-6 compressed problem is `PrimalInfeasible` and
no step completes.

## Files owned and changed

| file | change |
|---|---|
| `paper_5/src/paper5/stability/apcm_carried_witness.py` | `CWRMF_BUNDLE_SEED_POLICIES`, `DEFAULT_BUNDLE_SEED_POLICY`, `_bundle_growth_target`; `CWRMFSettings.bundle_seed_policy`, `.bundle_seed_spectral_threshold`, `.bundle_seed_rank_margin`, `.bundle_growth_factor`, `.bundle_growth_trigger`, `.bundle_interior_overshoot` with their validation; `CarriedWitnessModel._bundle_accepted_rank`, `._bundle_accepted_basis`, `.bundle_accepted_rank`, `._bundle_seed_working_set`; in `_solve_relaxed_readable_bundle_cone` the seed call, the compressed margin, the growth padding, the padding cursor reset on rank back-off, and recording the accepted working set |
| `paper_5/src/paper5/stability/apcm_carried_witness_analysis.py` | `--bundle-seed-policy`, `--bundle-seed-spectral-threshold`, `--bundle-seed-rank-margin`, `--bundle-growth-factor`, `--bundle-growth-trigger`, `--bundle-interior-overshoot`, each applied only when the caller passes it |
| `paper_5/tests/test_apcm_carried_witness.py` | 33 -> 41 tests: the defaults and their validation, the default seed being the literal leading-eigenvector block, each named seed policy's starting rank, the carried working set's span and its rank cap, the exit filter, the growth target arithmetic, the interior overshoot leaving the acceptance constants alone, and the parser defaults |

`apcm_positive_extension.py`, `cwrmf_solver_instances.py`,
`cwrmf_solver_replay.py`, `cwrmf_solver_replay_bench.py`,
`cwrmf_solver_instances_test.py` and `cwrmf_solver_replay_test.py` are in this
key's ownership list and were not touched.

Harnesses (under `output/local_runs/`, git-ignored):
`paper_v_bundle_rank_seed_20260915_v1/{rank_walk,seeded_probe,spectral_counts,atom_probe,atom_table,step_detail,compare_run,collect}.py`.

Impact analysis, read-only from the main checkout's index
(`node /Users/lordskyler/Holstein_test/.gitnexus/run.cjs impact <name>
--direction upstream --repo /Users/lordskyler/Holstein_test`; the index is 4
commits behind this worktree):

| symbol | risk | epistemic | impacted | upstream |
|---|---|---|---:|---|
| `_solve_relaxed_readable_bundle_cone` | LOW | exact | 3 | `_solve_relaxed_readable` -> `radial_atom` -> `integrate_cwrmf_ssprk2` |
| `radial_atom` | LOW | exact | 3 | `integrate_cwrmf_ssprk2` -> `apcm_carried_witness_analysis.main` |
| `CWRMFSettings` | LOW | exact | 5 | the analysis, mode-audit, campaign and stitch drivers and `stability/__init__` |

The new symbols `_bundle_growth_target` and `_bundle_seed_working_set` are not
in the index; `grep -rn --include='*.py' -e _bundle_seed_working_set -e
_bundle_growth_target -e bundle_seed_policy -e bundle_growth_factor -e
bundle_interior_overshoot paper_5 pipelines` finds them only in
`apcm_carried_witness.py`, `apcm_carried_witness_analysis.py` and
`test_apcm_carried_witness.py`.

One continuation note for a later reader: `restore_bundle_rank_hint` restores
the monotone rank hint across a checkpoint resume but there is no restore for
the accepted working set, so a resumed rollout under a carried policy starts
its first atom from the recorded rule.  That is the declared fallback, and it
costs one atom's walk.

### (c) part two: the balanced profile over `t=0` to `0.05` at `h=0.0025`

Balanced default first, as the comparison basis, same preparation and
coupling: 20 SSPRK2 steps, `--numerical-profile balanced --tier-timing`.
Artifact `output/local_runs/paper_v_cwrmf_balanced_default_t005_h0025_20260915_v1/`.

| quantity | balanced default, `t=0` to `0.05` |
|---|---:|
| autonomous wall seconds | `259.579` |
| Clarabel solve tier seconds / calls | `257.439` / 50 |
| all-coordinate scalar RMS against the exact cutoff-16 reference | `5.8368753441988074e-05` |
| maximum readable-rate residual / maximum completion correction norm | `2.9030e-05` / `2.9280e-05` |
| driver-process peak resident set, MiB | 1735 |

The balanced profile has almost nothing for this round's options to cut.
Reading the accepting iteration off `progress.jsonl`, 16 of the 21 bundle
atoms it reports accept on their **first** solve, 4 on their second and one
on its sixth: 9 discarded solves out of 30, against the strict rollout's 26
out of 33.  The balanced `psd_inflation = 1e-8` lifts the whole predictor
spectrum above the strict entry threshold, so the monotone hint arrives at a
working set that is already large enough, and the tier stops walking.

The best strict option does not transfer.  With
`carried_accepted_basis` and `bundle_seed_rank_margin = 2` the balanced
rollout inflates the working set instead of saving solves, because the cap is
relative to the previous accepted rank and therefore compounds by the margin
at every atom:

| time | balanced default: rank, step seconds | `carried_accepted_basis` margin 2: rank, step seconds |
|---:|---|---|
| 0.00250 | 33, 8.923 | 33, 8.972 |
| 0.00500 | 35, 7.726 | 35, 4.218 |
| 0.01000 | 35, 4.191 | 41, 9.732 |
| 0.01500 | 36, 4.699 | 49, 32.087 |
| 0.02000 | 37, 9.586 | 55, 72.852 |
| 0.02500 | 37, 5.400 | 39, 9.238 (rank back-off fired) |
| 0.03000 | 38, 11.151 | 47, 26.503 |
| 0.03500 | 39, 7.402 | 53, 66.454 |
| 0.04000 | 39, 7.394 | 34, 3.806 (second back-off) |
| 0.04500 | 39, 7.411 | 38, 6.960 |
| 0.05000 | 39, 7.392 | 45, 20.440 |

"Step seconds" is the run's `atom_s`, the maximum radial-atom wall time within
the step.  The balanced default settles at rank 39 and about `7.4 s` per step;
the carried policy climbs to rank 55 and `72.9 s` by `t=0.02`, is thrown back
to 39 by the tier's own rank back-off (a failed solve rebuilds the working set
from the predictor at half the rank), climbs again to 53, is thrown back
again, and finishes at 45.  Both rollouts reach `t=0.05` in 20 steps:

| quantity, `t=0` to `0.05`, balanced, `h=0.0025` | default | `carried_accepted_basis` margin 2 |
|---|---:|---:|
| Clarabel solve tier seconds / calls | `257.439` / 50 | `879.775` / 51 |
| autonomous wall seconds | `259.579` | `882.419` |
| all-coordinate scalar RMS | `5.8368753441988074e-05` | `5.836875342974831e-05` |
| maximum absolute difference of the 31 reported archive coordinates | -- | `4.309e-10` |
| maximum absolute difference of the 384 carried coordinates | -- | `1.124e-03` |
| maximum readable-rate residual | `2.9030e-05` | `2.8753e-03` |
| maximum completion correction norm | `2.9280e-05` | `2.0999e-03` |
| driver-process peak resident set, MiB | 1735 | 3876 |

Verdict for the balanced profile: dropped, with the measured reason rather
than a tuning failure.  The option pays only where the tier walks; the
balanced tier does not walk, so carrying the working set only enlarges it, at
`3.42x` the Clarabel time, two decades more readable-rate residual and
completion correction, and a peak resident set of `3876 MiB` against the
`4000 MB` guard.

### (c) part three: the strict rollout with the named `max_threads_8` profile

Same strict `t=0` to `0.01` rollout at `h=0.0025`.  `max_threads_8` is
round 2's named Clarabel setting set, off by default, qualified there against
the balanced full-Gram acceptance test and not bitwise.  The two options are
independent: one changes which working set the tier starts from, the other
changes how Clarabel factorises the resulting problem.

Speedups below are against the default-path bracket measured immediately
after this pair of runs, `152.563 s` of Clarabel tier; the four brackets of
this session are `152.774 s`, `152.448 s`, `152.563 s` and `150.703 s`, all
bitwise to the anchor.  The anchor's `138.644 s` was measured on a quieter
machine.

| run | calls | tier s | wall s | speedup | all-coordinate scalar RMS | archive | carried | readable | correction | peak MiB |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| default, bracket B | 36 | 152.563 | 153.850 | `1.00x` | `5.827261740618364e-05` | bitwise | bitwise | `1.4396e-07` | `2.9536e-06` | 1878 |
| `max_threads_8` alone | 37 | 102.568 | 103.927 | `1.49x` | `5.827261740618476e-05` | `2.089e-15` | `3.800e-06` | `2.9536e-06` | `1.0202e-05` | 2497 |
| `carried_accepted_basis` margin 2 alone | 27 | 104.227 | 105.712 | `1.46x` | `5.827261740618362e-05` | `1.443e-15` | `1.084e-06` | `3.2478e-07` | `2.9541e-06` | 1516 |
| **`carried_accepted_basis` margin 2 + `max_threads_8`** | **28** | **74.899** | **76.022** | **`2.04x`** | `5.8272617406184196e-05` | `1.742e-15` | `7.947e-07` | `4.2952e-07` | `2.9512e-06` | 2599 |
| `carried_accepted_basis` margin 0 + `max_threads_8` | 22 | 44.872 | 45.854 | `3.40x` | `5.827261740618511e-05` | `9.535e-14` | `2.171e-05` | `1.1781e-05` | `1.8980e-05` | 2471 |

The two accelerations compose: `1.46x` and `1.49x` separately, `2.04x`
together at margin 2 and `3.40x` together at margin 0.  The guarded
quantities of the margin-2 combination stay in the anchor's band -- minimum
carried unshifted 62-row Gram eigenvalue `-2.746834e-12` against the anchor's
`-3.888161e-12`, minimum carried shifted lower bound `6.242763e-12` against
`5.105582e-12`, both inside the strict acceptance band
`schur_safety_margin - 1e-13 = 1.9e-12` after the `1e-11` inflation; maximum
active critical modes 44 against 40, mean 29.0 against 27.8; maximum
readable-rate residual `4.2952e-07` against `1.4396e-07` and maximum
completion correction norm `2.9512e-06` against `2.9536e-06`.  The margin-0
combination's maximum readable-rate residual is `1.1781e-05`, above the
declared `readable_rate_tolerance = 1e-5`, which is the price of the extra
`1.4x`.

Two qualifications carried forward from round 2 apply unchanged to any run
that uses `max_threads_8`: it is not bitwise (6 of 36 recorded instances
agreed exactly with the pinned solutions, worst completion-velocity
difference `2.255e-04`) and it fails the strict full-Gram acceptance test on
one of the seven instances the pinned run accepts, so it is qualified against
the balanced acceptance test, not the strict one.  Promoting it needs D3.

## Tests

`cd paper_5 && PYTHONPATH=src python -m pytest -q -p no:cacheprovider` gives
**428 passed, 9 warnings in 187.08 s** under
`ram_guard.py --limit-mb 4000` (`peak=1972MB`).  The three files this key
owns and touches -- `test_apcm_carried_witness.py` (41, 33 before this
round), `cwrmf_solver_instances_test.py` (5) and
`cwrmf_solver_replay_test.py` (13) -- collect 59 and run in `3.22 s`.

The default-path gate is the rollout, not only the unit tests: with all six
new fields at their defaults the strict `t=0` to `0.01` rollout reproduces the
round-1 anchor `output/local_runs/paper_v_cwrmf_final_strict_t001_h0025_20260914_v1`
**bitwise** -- every array of `autonomous_trajectory.npz` and `trajectory.npz`
except `maximum_atom_seconds` equal under `np.array_equal`, worst difference
`0.000e+00`, all-coordinate scalar RMS `5.827261740618364e-05` to every
printed digit -- measured four times through the session, before the first
edit and after each edit batch.

## Kept

- Six named bundle seed policies on `CWRMFSettings.bundle_seed_policy` and the
  driver's `--bundle-seed-policy`, default `monotone_hint` which is the
  recorded rule and reproduces the anchor bitwise.
- `bundle_seed_spectral_threshold` (default `1e-6`) and
  `bundle_seed_rank_margin` (default 0), the declared parameters of those
  policies; at the defaults neither is read by the default policy.
- `bundle_growth_factor` (default `1.0`) and `bundle_growth_trigger`
  (default `0.0`): at the default factor the padding branch is never entered.
- `bundle_interior_overshoot` (default `1.0`): at the default the compressed
  base is built with exactly `schur_safety_margin`.
- The recommendation, off by default and not promoted:
  `--bundle-seed-policy carried_accepted_basis --bundle-seed-rank-margin 2`,
  `1.46x` on the strict `t=0` to `0.01` Clarabel tier alone and `2.04x` with
  `--solver-setting-profile max_threads_8`, with the 31 reported archive
  coordinates agreeing with the anchor to `1.742e-15` and every reported
  feasibility quantity in the anchor's band.

## Dropped

- Seeding the working set by rank alone.  `previous_accepted_rank` is
  bitwise identical to the default on this rollout; `spectral_threshold` at
  `1e-4` cuts the solve count from 36 to 14 and the time only from
  `152.563 s` to `130.913 s`, because the cost per solve rises as fast as the
  count falls; `previous_accepted_rank_margin` at margin 3 gives `1.14x`.
  The offline probe says why: a working set of the predictor's own lowest
  eigenvectors needs 14 more directions than the walked one at atom 1, and at
  seeded ranks 46 and 52 the compressed problem returns `NumericalError`.
- The enlargement growth factor.  At factor 1.5 with the default trigger the
  working set runs away to rank 61, Clarabel returns `NumericalError`, the
  completion correction norm rises to `1.0738e-03` and the rollout stops at
  `t=0.005` of `0.01` after `334.806 s`.  With the trigger raised to `1e-7`,
  so that only a badly failing solve grows, it completes at `1.08x` -- inside
  what one bracket pair of this machine's own spread would not cover, but not
  a lever.
- The compressed problem's own dual as an early acceptance test: refuted
  structurally and measured (all 26 discarded solves report compressed
  residuals at or below `4.916e-11` while their full-Gram minima reach
  `-5.576e-08`), and worthless even if it existed, since the full 62-row
  eigendecomposition that does certify costs `0.047 s` of `139.883 s`.
- `bundle_interior_overshoot`: at 3 the rollout takes 38 solves instead of 36
  at the same time; at 10 the first bundle atom's rank-6 compressed problem is
  `PrimalInfeasible` and no step completes.
- `carried_accepted_basis_exit` at both thresholds tried: `0.88x` at `1e-4`
  and `0.57x` at the default `1e-6`.  Dropping the risen carried modes keeps
  the working set small -- its maximum readable-rate residual is `1.9619e-07`,
  the lowest of any run in this round -- but the tier then has to rediscover
  the dropped directions, 65 solves against the default's 36.
- The carried working set in the balanced profile: `3.42x` slower over
  `t=0` to `0.05`, for the measured reason that the balanced tier accepts on
  its first solve at 16 of the 21 bundle atoms its progress log reports.

## Blockers and what this leaves for the next round

- The first bundle atom cannot be helped by anything in this round.  It has
  no previous working set to carry, its 13 solves cost `11.5 s` of the
  margin-2 run's `104.2 s`, and every static seed measured at that atom is
  slower than its walk.  A rollout long enough for that to matter does not
  exist yet.
- `carried_accepted_basis` is qualified only on the strict profile and only
  over `t=0` to `0.01`, eight radial atoms.  Its rank cap is relative to the
  previous accepted rank, so the working set can grow by the margin at every
  atom; over the balanced `t=0` to `0.05` rollout that compounding is what
  made it `3.42x` slower, held in check only by the tier's own rank back-off.
  A cap that does not compound -- tied to the endpoint spectrum rather than
  to the previous rank -- is the obvious next thing to try, and
  `carried_accepted_basis_exit` is the one form of it measured here, at
  `0.88x` and `0.57x`.
- The `2.04x` and `3.40x` combinations inherit `max_threads_8`'s
  qualification: balanced acceptance, not strict, and only on this 14-core
  M4 Pro.  Promotion needs D3.
- No rollout longer than `t=0.01` has been run in the strict profile with any
  of these options, so whether the carried working set keeps accepting on the
  first solve once the accepting rank approaches the critical mode limit 61 is
  unmeasured.  The strict rollout reached rank 43 at `t=0.01`.
