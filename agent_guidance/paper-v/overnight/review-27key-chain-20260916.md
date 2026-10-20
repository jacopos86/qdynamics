# Review of the 27-key energy-neutral chain (projective-route task 4)

Reviewer pass over `agent_guidance/paper-v/overnight/projective-route-20260915b.md:984-1327`
("(a) The chained rollouts", "(b) Does the energy-neutral repair remove the
phonon-occupation penalty?", "(a) continued: `t = 8` to `20`", "(c) Is it the
best Paper V moment-closure trajectory on record?"), committed as `0615282f`
with the worklog block `paper_5/notes/electron_phonon_closure_worklog_20260803.md:5272-5340`
while this review was being written.  Every number below was loaded from the
saved arrays or produced by a run made here; nothing was read back from a
`summary.json`, a score file or the report.  Scripts, logs and artifacts are
under `output/local_runs/paper_v_review_27key_chain_20260916/`
(`recompute_chain.py`, `rescore_comparators.py`, `archive_eom_t20.py`,
`inspect_artifacts.py`, `inventory_t8_t20.py`, their `.json` and `.log`
outputs, `exact_review_t20.npz`, the two `archive_eom_*_t20.npz` lanes, the
probe rerun, and `tests_head2d8f0efb/`).  The chain's own source,
`output/local_runs/paper_v_projective_dictionary_energy_20260915e/head_src/`,
is byte-identical to `paper_5/src` at commit `2d8f0efb` (checked with
`git archive` and `diff -rq`), and the worktree at review time differs from it
only in `cheap_projection_route.py` and `cheap_projection_route_analysis.py`,
which are the other key's since-committed edits.

Runs made here, each alone, heavy ones under `ram_guard --limit-mb 2000`:

| run | wall | peak |
|---|---:|---:|
| the five projective test files on the worktree | `86.63 s`, 85 passed | `477 MB` |
| the committed (`2d8f0efb`) `test_cheap_projection_route.py` against `head_src` | `8.57 s`, 28 passed | `450 MB` |
| chain recomputation with the exact cutoff-16 reference regenerated in process (8,001 samples) | exact `3.2 s` | `792 MB` |
| the 27-key extension-sensitivity probe, rerun from the record's script | --- | `626 MB` |
| the two archive-EOM review lanes to `t = 20` | `3.07 s` each | light |
| rescoring 16 comparator artifacts and chains | light | light |

Conventions used throughout, stated once: **all-coordinate scalar RMS** is
`sqrt(mean(d**2))` over the samples with `t <= T` inclusive on the `h = 0.0025`
grid from `t = 0` (1,601 / 2,401 / 3,201 / 4,801 / 6,401 / 8,001 samples at
`T = 4 / 6 / 8 / 12 / 16 / 20`) and all 31 closed scalar coordinates,
`d = approximate - exact`; every route here has a `t = 0` offset of
`8.08e-16`, so the raw and offset-subtracted figures coincide and only the raw
is printed.  Blocks are rho 0:3, B 3:7, N 7:11, A 11:17, C 17:31.
**Eq. (22) internal-energy RMS** is the sample RMS of
`matrix_total_energy(closed_scalar_to_matrix_state(row))` differences.
**site-0 occupation RMS** is the sample RMS of `rho[0,0]` differences, not
doubled, unless a column says otherwise.  The exact reference is
`exact_holstein_driven_trajectory` (DOP853, `rtol 1e-9`, `atol 1e-11`,
`max_step 0.01`) from the zero-field cutoff-16 ground state, regenerated here
on the 8,001 chain times: it is **bitwise equal** to the record's
`exact_t20.npz` and `exact_t8.npz` and to the stored `exact_archive_coordinates`
of all six `drive_power` segments; the four `post_pulse` segments' stored rows
differ from it by at most `3.1e-15` (their grids carry a prepended `t = 0`).

## Verdicts

| claim | verdict |
|---|---|
| (a) the six-segment `drive_power` chain, its per-segment ledger, the `t = 0` to `4` and `t = 0` to `8` tables, the cumulative table to `t = 20`, the chain totals | **pass on every number** --- reproduced to all printed digits from the arrays (tables below); the segments tile with no gap, overlap or discarded sample |
| (b) the constraint reverses the phonon-occupation penalty; the block table; the build-time sensitivity table | **pass** --- every ratio recomputes; the probe rerun on the committed tree reproduces the stored 29-row table **bitwise** |
| the energy is held on its reference; the reference carries no exact information | **pass** --- `max |E - E_ref| = 1.78e-15` over all 8,001 samples, and the route's `internal_energy` equals `matrix_total_energy` of its own archive rows to `1.78e-15`; the reference is built from the closure's own velocity at the propagated state (code cited below) |
| the comparators cell A `0.012580`, both cones `0.128607` / `0.000939628`, plain archive EOM `0.193205` / `0.0182361` | **pass on the quoted values**; **fail on the site-0 occupation column** of the `t = 20` table, which sets this chain's `rho[0,0]` sample RMS beside the record's `2 rho[0,0]` trapezoidal time-RMS --- a factor 2 --- and the ordering in that column reverses in one convention |
| (c) "the best Paper V moment-closure trajectory on record ... at every horizon it reached" | **fail at `t = 20`, and the "best other row" column is wrong at `t = 12` and `16`.**  The route's own second variant, the 27-key `post_pulse` chain, whose segments 2 to 4 are on disk from the same round's queue and are not scored in the record, reads `6.990927e-02` over `t = 0` to `20` against this chain's `8.519313e-02`, at `3,886.9 s` against `4,794.4 s` |
| "on accuracy per unit of wall ... this chain wins" | **unmeasured**: `5,832.34 s` is the previous machine with four Clarabel/faer threads; `4,794.4 s` is this machine with other agents' jobs possibly present, as the report itself says |

---

## 1. The chain reproduces

### Tiling

Each resumed segment's first sample is the previous segment's resume sample:
at all five `drive_power` boundaries (`t = 3.96`, `7.84`, `8.0`, `11.84`,
`16.32`) the 56-coordinate state, the 31 archive coordinates and the 70
standardized frontier coordinates are `np.array_equal`, the `resume.npz`
state equals that row, the time gap is exactly `0.0`, the Eq. (22) energy and
its reference are continuous to `8.9e-16`, and every segment's completed step
count (1,584 / 1,552 / 64 / 1,536 / 1,792 / 1,472) is a multiple of 16 and
equals its resume index, so **no sample was discarded** and the completion is
re-solved on the steps a single rollout would have used.  The concatenated
grid, dropping each duplicate boundary row, is 8,001 samples on
`k * 0.0025` to `3.6e-15`, strictly increasing.  The `post_pulse` chain
(task-3 rollout at index 1,600, then segments 1 to 4 at `t = 4.0`, `8.0`,
`12.6`, `17.48`) tiles the same way, 1,600 / 1,600 / 1,840 / 1,952 / 1,008
steps.

### Scores, `drive_power`, cumulative from `t = 0`

| `T` | all 31 | 27 outside N | N (4) | rho / B / A / C | Eq. (22) energy | site-0 (`rho[0,0]`) | min retained joint Gram | min scaled extension | max envelope |
|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 4 | `1.142716e-02` | `1.164437e-02` | `9.836299e-03` | `2.0368e-2` / `6.9088e-3` / `8.9418e-3` / `1.1167e-2` | `4.597225e-03` | `1.430879e-02` | `-4.535e-04` | `-4.852e-05` | `0.3694` |
| 6 | `2.090571e-02` | `1.856285e-02` | `3.257634e-02` | `2.9279e-2` / `1.1316e-2` / `1.6688e-2` / `1.8025e-2` | `5.394635e-03` | `2.116592e-02` | `-1.174e-03` | `-1.442e-04` | `0.3937` |
| 8 | `2.593079e-02` | `2.474607e-02` | `3.282777e-02` | `4.0173e-2` / `2.7248e-2` / `1.9942e-2` / `2.1275e-2` | `5.751890e-03` | `3.027929e-02` | `-2.152e-03` | `-1.922e-04` | `0.4518` |
| 12 | `4.250925e-02` | `4.385377e-02` | `3.198825e-02` | `6.9808e-2` / `7.5186e-2` / `3.0539e-2` / `2.5494e-2` | `6.088286e-03` | `5.278107e-02` | `-2.463e-03` | `-2.040e-04` | `0.4518` |
| 16 | `6.377276e-02` | `6.722257e-02` | `3.188383e-02` | `1.0303e-1` / `1.3219e-1` / `3.5397e-2` / `3.0187e-2` | `6.249722e-03` | `8.393451e-02` | `-2.463e-03` | `-2.040e-04` | `0.4518` |
| 20 | `8.519313e-02` | `9.045792e-02` | `3.187001e-02` | `1.3580e-1` / `1.8729e-1` / `4.3970e-2` / `3.1283e-2` | `6.344620e-03` | `1.125899e-01` | `-2.463e-03` | `-2.040e-04` | `0.4518` |

Every entry the record prints at these horizons is reproduced to its last
digit, including the `t = 4` and `t = 8` Euclidean differences
(`1.551037e-01` / `1.439192e-01`; `2.372364e-01` / `2.292268e-01`), the
27-key unconstrained row (`1.630660e-02`, N `2.974702e-02`, energy
`7.088814e-02`, site-0 `1.007098e-02`, `-1.877e-08` / `-6.347e-06`, envelope
`0.3646`, `2.350741e-01` / `1.829419e-01`, `749.1 s`, 100 solves / 18
refused at `7.10 s` mean) and the `post_pulse` `t = 0` to `8` row
(`4.537517e-02`, `4.219396e-02`, `6.276381e-02`, `1.373215e-01`,
`3.984892e-02`, `-2.907e-04`, `-2.486e-05`, `0.3748`, `1,429.0 s`, 200 / 30).
The repository scorer `_accuracy_metrics` on the same rows agrees with the
direct computation on every key used.

### Ledger and totals

Per segment (`drive_power` 1 to 6): steps 1,584 / 1,552 / 64 / 1,536 / 1,792 /
1,472; autonomous wall `1,011.7` / `1,010.8` / `35.5` / `1,000.4` / `1,004.5` /
`731.5 s`; completions re-solved / refused 99/32, 97/42, 4/1, 96/32, 112/29,
92/8; wall in those solves `939.3` / `937.6` / `32.4` / `926.6` / `919.9` /
`661.1 s`; repairs 4,711/41, 4,656/0, 192/0, 4,608/0, 5,376/0, 4,416/0, all
constrained, 0 hyperplane refusals; peak `ru_maxrss` 630 / 626 / 628 / 611 /
631 / 629 **MiB** (the report's "MB" column is `bytes / 2^20`).  Totals:
8,000 steps, `4,794.4 s`, 500 solves / 144 refused, `4,416.9 s` in solves
(`8.83 s` mean, 92.1 percent), 23,959 repairs applied / 41 skipped,
`250.5 s` repair, `124.8 s` velocity, `max |E - E_ref| 1.78e-15`, peak
`631 MiB`; to `t = 8`: `2,058.0 s`, 200 / 75, 9,559 constrained repairs; to
`t = 4`: `1,022.1 s` by the report's own rule.  All as printed.

### The energy hold and the `drive_power` reference

`internal_energy - internal_energy_reference` is at most `1.776e-15` over the
8,001 chain samples, and `internal_energy` equals `matrix_total_energy` of
the chain's own archive rows to `1.776e-15`, so the hyperplane the repair
holds is the scorer's Eq. (22) functional.  The reference is `-3.7753265535`
at `t = 0` (the closure's own energy of the prepared state, equal to the exact
ground energy), `-3.5646834904` at `t = 4`, `-3.5646877627` at `t = 20`,
moving `5.2e-6` over `t = 4` to `20` (the exact Eq. (22) energy moves
`3.07e-6` there and `0.28691` over the drive).  Against the exact
`-3.5713979820` at `t = 4` the final reference sits `0.006710` above, as the
report says; the value it prints, `-3.564687762684483`, is the reference
**after** the drive tail (`t >= 7.84`), not at `t = 4` (`0.006714` above
there).  The 15-key `drive_power` artifact's reference is `-3.5441711275`
at `t = 4` and `-3.5441789365` at `t = 20`, `0.027227` and `0.027219` above
the exact value; the report's `0.027219` is the `t = 20` figure.

**No exact information enters the propagation.**  In the committed
`cheap_projection_route.py` (`head_src`, byte-identical to `2d8f0efb`) the
`drive_power` reference starts at `energies[0]`, the closure's own Eq. (22)
value of the initial or resumed state, and is advanced with the SSPRK(3,3)
weights by `energy.gradient @ (v(t, x, lift) - v(0, x, lift))`, both velocities
being `route.unguarded_velocity` at the **propagated** state (the `velocity`
closure inside `integrate_cheap_projection_ssprk3`, and the three stage
updates at the `first_reference` / `second_reference` / `final_reference`
lines).  `DimerParameters.drive_difference(0) = 2 A sin(0) exp(0) = 0`
exactly, so `v(0, x)` is the drive-free velocity of the same coordinates.
`cheap_projection_route.py` imports nothing from `exact_reference`; the only
exact call in the propagation modules is `exact_holstein_joint_moment_initial_state`
at `projected_apcm.py:1528`, the `t = 0` preparation.  `run_energy_segment.py`
calls that preparation (through the retry ladder) even on a resumed segment,
then overwrites the state and frontier from `resume.npz` before integrating,
and builds the exact reference only after the rollout returns.  Confirmed.

## 2. Section (b)

The phonon-occupation ratios recompute: `9.836299e-03 / 2.974702e-02 = 0.331`,
`/ 1.589107e-02 = 0.619`, `2.974702e-02 / 1.589107e-02 = 1.872`; the block
ratios `1.357` / `1.526` / `0.331` / `0.567` / `0.843` / `0.701` / `0.882` /
`1.421` / `0.065`; the net `1.630660 / 1.142716 = 1.427`; the energy `15.4`.
The probe rerun on the committed tree reproduces the stored
`extension_sensitivity_27key.json` with **zero** difference on all 29
extension and retained derivatives: Bloch `4.691523` / `4.472911` /
`4.691523`, `raw_normal_00` and `_11` `0.146282`, `raw_normal_01_real`
`0.292565`, `raw_normal_01_imag` exactly `0`; the 15-key file reads
`4.690417` / `4.472137` / `0.104966` / `0.209933` / `0`; ratios `3.12` to
`6.54` percent at 27 keys, `2.24` to `4.48` at 15, enlargement factor
`1.3935`.  Naming: "the retained cone sees all 29 raw coordinates equally"
--- the retained derivatives take two values, `1.414214` on the eight
diagonal normal and anomalous coordinates and `2.000000` on the other 21;
within a factor `1.41`, not equally.

## 3. The comparators and their conventions

| comparator as quoted | where it is defined | convention | quoted correctly? |
|---|---|---|---|
| cell A `0.012580`, site-0 `0.006957`, static energy `7.963e-4`, retained `2.889e-5`, extended `M4` `-8.76e-7`, `3,399.6 s`, `t = 0` to `4` | worklog:2608-2616 | `_trajectory_metrics`, `adaptive_positive_moment_analysis.py:78-135`: raw all-coordinate scalar RMS, `rho[0,0]` sample RMS, `matrix_total_energy` sample RMS --- **the same three conventions as this chain's** | yes; the construction differs (60 coordinates, `PositiveFourthMomentCompletion` at `logdet_shift 1e-5`), as the report says; artifact not on disk |
| both cones `0.128607`, internal energy `9.39628e-4` (`0.000939628`), "total site-0 occupation RMS error" `0.204690`, retained `1.36819e-5`, `5,832.34 s`, `t = 0` to `20` | worklog:2783-2792 and 2898-2903 | the progression scorer `pipelines/open_dynamics/plot_paper_v_results_progression.py`: `_time_rms` is **trapezoidal in time** (`:375`), and its occupation observable is **`2 rho[0,0]`** (`_observables`, `:308`) | score and energy yes (sample against trapezoidal differs by `4e-5` relative on this grid: plain lane `0.01823533` against `0.01823608`); **site-0 no**, see below |
| plain archive EOM `0.193205` / `0.0182361` / `0.286170`, `t = 0` to `20` | worklog:2900-2903 | same scorer; the lane is `closed_scalar_rhs` at `dt = 0.01` (the audit's `0.11066` against `0.11089` at `t = 4`) | yes as a record line; at `h = 0.0025` the same lane reads `0.193196` / `0.018235` / `0.143113` (`rho[0,0]`) here |

**The site-0 occupation convention, established from the plain archive EOM
lane at `t = 20` (`archive_eom_t20.json`):** the record's `0.286170` is
reproduced as `0.2861692` by `2 rho[0,0]` trapezoidal time-RMS (`2 rho[0,0]`
sample RMS `0.2862254`), while `rho[0,0]` sample RMS reads `0.1431127`; the
record's `0.0182361` is reproduced as `0.01823608` by the trapezoidal energy
RMS.  So the record's both-cones `0.204690` and plain-archive `0.286170` are
`2 rho[0,0]` trapezoidal time-RMS, and the chain's `1.125899e-01` is
`rho[0,0]` sample RMS.  In the record's convention the chain reads
**`0.22518`** at `t = 20` (`0.02862` at `t = 4`), which is *worse* than both
cones' `0.204690`, not better; the `t = 20` table's site-0 column
(`projective-route-20260915b.md:1274-1279`) reverses its ordering when put in
one convention.  The same crossing is in the cheap-projection report and its
worklog blocks wherever a cheap-route site-0 figure (`0.199933`, `0.203657`,
`0.186909`, `0.247970`, `0.210425`, `0.212252`, `0.290898`, `0.112590`) is set
beside `0.204690` or `0.286170`: `cheap-projection-20260915b.md:332, 574,
615-617, 783, 965, 1037-1038, 1153-1157, 1270, 1310, 1472` and
`worklog:4864, 4960-4961, 5137-5138`.  The conventions audit's table (a),
fact 2, named this class and said no 2026-09-14/15 block crossed it; these
blocks do.  The `t = 4` table is unaffected (cell A is `rho[0,0]`).

## 4. Section (c): the "best on record" claim

### The route's own second variant, scored to `t = 20`

`paper_v_dictionary_energy_postpulse_27key_n16_20260915e_seg2` to `_seg4`
were produced by the report's own `queue_remaining.sh` at 16:58, 17:15 and
17:22 on 2026-09-15, reach `t = 12.6`, `17.48` and `20.0`, and are neither
scored in the report nor listed in its provenance (which names `seg{0,1}`).
Same configuration as the chain under review except the reference mode.
Cumulative from `t = 0`, one convention, both chains:

| `T` | `drive_power` all 31 | `post_pulse` all 31 | `drive_power` Eq. (22) | `post_pulse` Eq. (22) | `drive_power` site-0 (`rho[0,0]`) | `post_pulse` site-0 | `post_pulse` N (4) | `post_pulse` min retained / extension |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 4 | `1.142716e-02` | `1.630660e-02` | `4.597225e-03` | `7.088814e-02` | `1.430879e-02` | `1.007098e-02` | `2.974702e-02` | `-1.877e-08` / `-6.347e-06` |
| 6 | `2.090571e-02` | `3.335615e-02` | `5.394635e-03` | `1.193549e-01` | `2.116592e-02` | `3.325676e-02` | `4.396934e-02` | `-1.880e-04` / `-1.461e-05` |
| 8 | `2.593079e-02` | `4.537517e-02` | `5.751890e-03` | `1.373215e-01` | `3.027929e-02` | `3.984892e-02` | `6.276381e-02` | `-2.907e-04` / `-2.486e-05` |
| 12 | `4.250925e-02` | **`6.075829e-02`** | `6.088286e-03` | `1.531984e-01` | `5.278107e-02` | `4.751135e-02` | `8.295406e-02` | `-3.124e-03` / `-3.429e-04` |
| 16 | `6.377276e-02` | **`6.385168e-02`** | `6.249722e-03` | `1.605503e-01` | `8.393451e-02` | `4.662803e-02` | `8.798803e-02` | `-3.124e-03` / `-3.429e-04` |
| 20 | `8.519313e-02` | **`6.990927e-02`** | `6.344620e-03` | `1.648045e-01` | `1.125899e-01` | `6.249305e-02` | `8.256954e-02` | `-3.124e-03` / `-3.429e-04` |

`post_pulse` chain totals: `749.1 s` (the reused task-3 rollout) plus
`679.8` + `1,005.4` + `1,000.2` + `452.4 s` = **`3,886.9 s`** over `t = 0` to
`20`; 500 completions re-solved, 77 refused; 19,195 constrained repairs over
`t = 4` to `20` with `max |E - E_ref| 1.78e-15`; peak `713 MiB` (segment 4);
`post_pulse` blocks at `t = 20`: rho `8.031e-2`, B `1.1223e-1`, N `8.257e-2`,
A `7.062e-2`, C `4.190e-2`.

So on the 31-coordinate RMS the `drive_power` chain is the best row on record
at `t = 4`, `6`, `8` and `12`, ties at `t = 16` (`0.12` percent apart), and
is **not** the best at `t = 20`: its own `post_pulse` variant is `1.22` times
more accurate there at `0.81` of the wall.  On the Eq. (22) energy the
ordering is the reverse by a factor 26 (`6.34e-3` against `1.65e-1`), and the
`post_pulse` chain's cones are further violated (`-3.12e-3` / `-3.43e-4`
against `-2.46e-3` / `-2.04e-4`).  The corrected "best other row on record"
column of the (c) table:

| horizon | this chain | best other row, corrected | as printed |
|---:|---:|---|---|
| `t = 4` | `1.142716e-02` | cell A `1.258000e-02` | same |
| `t = 8` | `2.593079e-02` | 15-key `post_pulse` `N = 16` `4.254184e-02` (27-key `post_pulse` `4.537517e-02`; 15-key `N = 4` post-pulse rows `4.672e-2` to `4.697e-2`) | `4.254200e-02`, same row |
| `t = 12` | `4.250925e-02` | 27-key `post_pulse` `6.075829e-02` | 15-key `drive_power` `1.251390e-01` |
| `t = 16` | `6.377276e-02` | 27-key `post_pulse` `6.385168e-02` | 15-key `post_pulse` `1.950960e-01` |
| `t = 20` | `8.519313e-02` | **27-key `post_pulse` `6.990927e-02`**, then both cones `1.286070e-01` | both cones |

### The other rows on record, rescored here on the same reference

Every 15-key figure the (c) table and the `t = 0` to `8` / `t = 0` to `20`
tables quote reproduces from the cheap-route artifacts to the six digits
printed: `N = 16` `post_pulse` `4.254184e-02` / `1.384930e-01` /
`1.950965e-01` / `2.391718e-01` at `t = 8` / `12` / `16` / `20`;
`drive_power` `4.557070e-02` / `1.251386e-01` / `2.584988e-01` /
`3.286739e-01`; unconstrained `5.773004e-02` / `1.360163e-01` /
`3.235360e-01` / `5.793324e-01`, with N block `3.883838e-02` at `t = 6` and
`1.362353e+00` at `t = 20`; energies `6.060306e-02` / `2.405480e-02` /
`2.269427e-01` at `t = 8` and `6.938377e-02` / `2.599977e-02` /
`2.593707e+00` at `t = 20`; the `t = 4` rows `2.374447e-02` /
`2.684227e-02` with their N, energy, site-0, block, cone, envelope and
Euclidean entries; walls `669.8` / `719.6` / `486.2 s`.  No other artifact on
disk beats the `drive_power` chain at `t = 8` or `12`: the 15-key `N = 4`
chains read `4.685e-2` / `4.672e-2` / `4.697e-2` at `t = 8` and
`1.460e-1` / `1.464e-1` at `t = 12`; the frozen-lift Dykstra route
`2.839e-1` at `t = 8`; the carried-witness Dykstra route `7.833e-1`.  The
two closed-form 31-coordinate lanes the conventions audit introduced, run to
`t = 20` here at `h = 0.0025`: plain archive EOM `1.381741e-01` at `t = 8`
and `1.931960e-01` at `t = 20`; Eq. (112) archive EOM `8.833624e-02` and
`1.841194e-01` --- both worse than either 27-key chain at every horizon from
`t = 4`.  The recorded cone routes' artifacts (`paper_v_apcm_spin_exchange_blocks_*`,
cell A) are not under `output/local_runs`, as the report says.

### Wall

`4,794.4 s` against `5,832.34 s` is this machine against the previous one
(the both-cones run used four bounded Clarabel/faer threads, worklog:2770),
and the report's own provenance says other agents' jobs may have been on the
machine.  The `1.51x` at `0.82` of the wall is arithmetic on two clocks.

## 5. Gates and tests

| gate as written | here |
|---|---|
| chaining reproduces a single rollout: `1.732e-14` on the states and archive coordinates, same 33 sample times, reference to `8.9e-16` | states and archive `1.732e-14`, frontier `2.28e-15`, times to `1.4e-17`, `A[16] == B[0]` exactly; the reference over the 16 resumed samples agrees to **`1.33e-15`** --- the `8.9e-16` is segment B's own `maximum_absolute_deviation_from_reference`, not the chain-against-single figure |
| a segment resumes on a re-selection boundary | every resume index is the segment's completed step count and a multiple of 16 |
| the recorded driven interval is reusable for `post_pulse` | `np.array_equal` on times, states, frontier, both eigenvalue series and archive coordinates over the 81 samples, as stated; the stored exact rows differ by `4.4e-16` (not claimed) |
| the constraint is off before its declared time | **not a gate**: the `post_pulse` chain's `t = 0` to `4` *is* the task-3 rollout, seeded from its index-1,600 row by `seed_resume_from_run.py`; the sentence compares an artifact with itself.  The gate that does exist is the 81-sample prefix rerun above |
| the energy is held | `1.776e-15` over 9,559 (`t = 0` to `8`) and over all 23,959 constrained repairs; `1.776e-15` over the `post_pulse` 4,800 |
| the five projective test files | 85 passed in `86.63 s`, peak `477 MB`, on the worktree at review time |
| the cheap route's tests | the committed test file (28 tests at `2d8f0efb`) passes against `head_src` in `8.57 s`; the worktree's file has since gained 6 tests from the other key and was not the chain's gate |

## 6. Smaller

- `projective-route-20260915b.md:1104`: the 15-key unconstrained `t = 0` to
  `4` wall is `89.6 s` in its artifact (`89.621 s`) and in the cheap report;
  `93.1 s` is not any recorded clock.
- The segment table's "peak resident ... MB" values are `ru_maxrss / 2^20`,
  i.e. MiB (`660,488,192` bytes = `630 MiB` = `660 MB`).
- The reported `drive_power` reference `-3.564687762684483` is the value from
  `t = 7.84` on; at `t = 4` it is `-3.564683490374372`.
- Provenance caution: `run_energy_segment.py` under the report's script
  directory was modified on disk while this review was being written
  (another key added an `extension_settings` argument to
  `prepare_with_ladder`, defaulting to the old behaviour); the directory is
  untracked, so the script that produced the segments survives only as the
  configuration blocks in their `summary.json`, which this review verified
  against.
- Reading 3 of "(a) continued": `0.12`-percent tie at `t = 16` with the
  `post_pulse` chain belongs beside "the blocks that grow are rho and B",
  since the `post_pulse` chain's rho and B at `t = 20` (`8.03e-2`, `1.12e-1`)
  grow less than this chain's (`1.36e-1`, `1.87e-1`) while its N block stays
  `2.6` times larger.

## Corrections for the worklog block `:5272-5340` and the report

1. **Withdraw "best ... at every horizon it reached" and the `0.662` /
   `0.82` reading.**  At `t = 20` the 27-key `post_pulse` chain (same
   dictionary, cadence and repair; reference held at the trajectory's own
   `t = 4` energy `-3.3905733452`), segments on disk under
   `paper_v_dictionary_energy_postpulse_27key_n16_20260915e_seg1` to `_seg4`,
   scores `6.990927e-02` over `t = 0` to `20` against this chain's
   `8.519313e-02`, at `3,886.9 s` against `4,794.4 s`, with Eq. (22) energy
   `1.648045e-01` against `6.344620e-03` and cones `-3.124e-03` /
   `-3.429e-04` against `-2.463e-03` / `-2.040e-04`.  Restate: best on record
   on the 31-coordinate RMS at `t = 4`, `6`, `8`, `12`; tied at `t = 16`
   (`6.377276e-02` against `6.385168e-02`); second at `t = 20`.
2. Correct the (c) table's "best other row" at `t = 12` and `16` to the
   27-key `post_pulse` chain's `6.075829e-02` and `6.385168e-02`, and at
   `t = 20` to `6.990927e-02` (both cones `1.286070e-01` third); add the
   `post_pulse` chain's `t = 12` / `16` / `20` rows to the cumulative table
   and its segments 2 to 4 (`t = 8` to `12.6`, `12.6` to `17.48`, `17.48` to
   `20`; 1,840 / 1,952 / 1,008 steps; `1,005.4` / `1,000.2` / `452.4 s`;
   115/16, 122/28, 63/3 solves; peaks 606 / 644 / 713 MiB) to the ledger and
   provenance.
3. **Put the `t = 20` site-0 occupation column in one convention.**  The
   record's both-cones `0.204690` and plain-archive `0.286170` are
   `2 rho[0,0]` trapezoidal time-RMS (progression scorer, `_observables`
   doubles, `_time_rms` is trapezoidal; the plain lane reproduces `0.286170`
   as `0.2861692` in that convention and reads `0.143113` in this chain's);
   this chain's `1.125899e-01` is `rho[0,0]` sample RMS and reads `0.22518`
   doubled, worse than both cones, not better.  Either double the chain's and
   the 15-key rows' site-0 figures in that table or halve the record's, and
   say which.  Apply the same to the cheap-projection report and worklog
   lines listed in section 3.
4. Say that `4,794.4 s` and `5,832.34 s` are two machines (four Clarabel/faer
   threads on the old one); the wall ratio is not measured.
5. `93.1 s` -> `89.6 s` for the 15-key unconstrained `t = 0` to `4` row.
6. "peak resident 631 MB" -> MiB (`bytes / 2^20`), in the report and the
   worklog block.
7. The gates row "the drive-power reference agrees to `8.9e-16`": the
   chain-against-single reference agreement over the 16 resumed samples is
   `1.33e-15`; `8.9e-16` is segment B's own hold deviation.  Drop the "the
   constraint is off before its declared time" row or restate it as the
   81-sample prefix rerun, since the `post_pulse` chain's `t = 0` to `4` is
   the task-3 artifact by construction.
8. "the retained Gram sees all 29 equally" -> "to within a factor `1.41`
   (`1.414` on the eight diagonal normal and anomalous coordinates, `2.000`
   on the rest)".
9. State that `-3.564687762684483` is the reference from `t = 7.84` on
   (`-3.564683490374372` at `t = 4`); both are `0.0067` above the exact
   `-3.5713979820`.
