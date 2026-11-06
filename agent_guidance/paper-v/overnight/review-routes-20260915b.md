# Review of three 2026-09-15 reports: cheap projection, projective route, CWRMF/CHTC lane

Reviewer pass over `agent_guidance/paper-v/overnight/cheap-projection-20260915b.md`,
`projective-route-20260915b.md` and `cwrmf-chtc-20260915b.md`, and over their
worklog blocks at `paper_5/notes/electron_phonon_closure_worklog_20260803.md:4595-4775`
(the two blocks written after the conventions-audit corrections at `:4526-4594`).

Every number below was loaded from the saved arrays or produced by a run made
here; nothing was read back from a report. Review scripts, logs and artifacts
are under `output/local_runs/paper_v_review_routes_20260915b/`
(`verify_affine.py`, `verify_dykstra_ladder.py`, `compare_wallbudget.py`,
`check_resume_table.py`, `check_negative_result.py`, `check_c_run.py`,
`inspect_margin.py`, the six `run_*.sh` wrappers that carry the exact command
lines, and the run directories `cw_dykstra30_t4/`,
`defaultgate_t005_worktree/`, `wb_unbudgeted/`, `wb_budget600/`). The source
overlay at `db1498b2`, the source overlay at `4510bd01` and the extracted
bundle were deleted after use; the commands that rebuild them are
`git archive <commit> paper_5/src | tar -x -C <directory>` and
`tar -xzf output/local_runs/paper_v_cwrmf_event_guard_accel_20260915/paper_v_cwrmf_event_guard_accel_20260915.tgz`.

Six runs were made, each alone and wrapped in
`pipelines/shell/ram_guard.py --limit-mb 2000`:

| run | wall | peak |
|---|---:|---:|
| carried-witness no-guard flow, Dykstra 30 per step, `t = 0` to `4` at `h = 0.0025` | `67.59 s` | `726 MB` |
| the three affine-Gram reconstructions and the 24-row design-matrix rank | `1.6 s` | `2 MB` |
| the Dykstra iteration ladder on the perturbed preparation | `9 s` | `224 MB` |
| the projective default path, `t = 0` to `0.05` at `h = 0.0025`, from the current worktree | `50.57 s` | `253 MB` |
| the seven resume and wall-budget tests | `64.24 s` | `1,262 MB` |
| the balanced event guard at `t = 0` to `0.05`, unbudgeted and with `--maximum-wall-seconds 600` | `2.57 s` + `2.6 s` | `651 MB` |

The cheap-projection rerun was made against the source at commit `db1498b2`
extracted into the review directory, because three other keys have since left
uncommitted edits in `paper_5/src/paper5/stability/` in this shared worktree.

## Verdicts

| report | verdict |
|---|---|
| `cheap-projection-20260915b.md` | **pass on every number checked** — the carried-witness rerun reproduces the artifact bitwise, and the affine-map claim holds at probes of my own. Three corrections: one omitted reported quantity that reverses the route's reading on the Eq. (22) internal energy, two comparisons made across the raw and offset-subtracted conventions, and two tables with no artifact. |
| `projective-route-20260915b.md` | **fail on two stated quantities, both the same mis-read.** The exact-start run took **31** stage retractions over its 394 accepted steps, not 2, and five of them fall inside the `t = 5.085` to `5.235` interval the report says the exact start passes "without a single retraction"; the step-halved (c) run took **64**, not 6. In both cases the printed number is the last recorded step's own count. A third (c) error: the retained-Gram row compares two different summary keys. Everything else checked reproduces, including the bitwise default-path gate, which I reran. |
| `cwrmf-chtc-20260915b.md` | **pass on every gate rerun here** — seven tests pass, the bundle digest and byte count are exact, the eight submit files validate. Two defects the report does not carry: the `wall_budget` resume the wall budget exists to enable cannot be executed with the shipped submit files as written, and the bundle pins an uncommitted working tree rather than a commit. |

---

## 1. `cheap-projection-20260915b.md`

### What reproduced

The carried-witness no-guard flow with Dykstra 30 applied to the 62-row
unified Gram after every accepted step, `t = 0` to `4` at `h = 0.0025`, strong
coupling (`lambda_ep = 1.5`, `gamma = 0.5`, drive amplitude 1), phonon cutoff
16, exact-correlated preparation, was rerun here. Every recorded number is
**bitwise** the artifact's:

| quantity, `t = 0` to `4`, 1,601 samples | rerun here | artifact |
|---|---:|---:|
| all-coordinate scalar RMS on the 31 closed scalar coordinates, raw | `0.18429968180952563` | identical |
| the same, offset-subtracted | `0.18430032941450072` | identical |
| `t = 0` coordinate offset, Euclidean norm | `3.2442360141805306e-04` | identical |
| minimum scaled 62-row unified Gram eigenvalue | `-3.6614666748693155e-05` | identical |
| the same before each repair | `-6.49177352457415e-04` | identical |
| minimum unshifted retained joint Gram eigenvalue | `-2.4989229334100913e-16` | identical |
| site-0 occupation RMS (`rho[0,0]`, not doubled) | `6.994750764087457e-03` | identical |
| blockwise scalar RMS, rho / B / N / A / C | `1.2754e-2` / `2.9197e-2` / `4.9933e-1` / `8.9174e-2` / `1.6989e-2` | identical |
| autonomous wall / propagation | `67.59 s` / `25.64 s` | `69.32 s` / `26.21 s` |

Recomputed independently from `trajectory.npz` rather than from
`summary.json`: `sqrt(mean(d**2))` over the 31 coordinates and 1,601 samples
gives `0.18429968180952563`, and with the run's own `t = 0` offset removed
`0.18430032941450072`. Both print as `0.18430`.

The control's `dynamic_all_coordinate_scalar_rms_error` is
`0.4804854378166632` and its raw value `0.48048030541364645`. The conventions
audit's own independent rerun of the record's no-guard flow through the
carried-witness driver
(`output/local_runs/paper_v_review_cwrmf_noguard_t4_h0025_20260915_v1`, table
in `output/local_runs/paper_v_review_conventions_20260915/one_convention_table.json`)
gives `0.4804854378166632` and `0.48048030541364645`, so the cheap route's
per-step call pattern of `integrate_cwrmf_ssprk2` is not merely equal to the
record's propagation to the printed digits — it is bitwise equal, on the
coordinate RMS, the C block (`0.4723429708234574`), the site-0 occupation
(`0.04864262523553689`) and the Eq. (22) internal energy
(`0.013879216451647403`). **Claim confirmed, and stronger than stated.**

The exact reference arrays of the cheap route's `t = 4` runs and of the
projective `t = 4` event-guard run are equal under `np.array_equal`, so every
`t = 0` to `4` comparison the report makes is valid sample by sample.

Every other table row I checked reproduces from its artifact:
the `t = 0` to `0.05` policy table (control `1.2719373770656832e-05`, N block
`2.9226e-08`, 24-row minimum `-1.644735e-11`, retained `2.8920804e-05`, 0/60
repairs, `0.141 s`; and the clip, Higham, Dykstra 5, Dykstra 30 and per-step
rows with their repair counts 46/14, 54/6, 46/14, 46/14, 15/5); the `t = 0` to
`4` fixed-dictionary table (`0.13304` / `-1.5159` / `-1.5578` / `0.4216`,
`0.13460` / `-2.2815e-3` / `-2.3217e-4`, `0.13134` / `-1.0431e-5` /
`-7.9239e-7` / `0.4487`, `frontier_only` `0.13489` / `-0.9211` / `-2.4067` /
`4.874`); `t = 0` to `0.015` (`1.7230549791672775e-06`, N `4.2818e-09`); `t = 0`
to `0.5` (`2.0130026379320347e-03`, `-2.977e-07`, `-2.890e-09`); and the
Dykstra 200 row (`0.18408611981799747`, `-4.246139e-07`, `171.4 ms` mean
repair). The record lines quoted are the artifacts': the no-retraction
ablation is `1.9407597628467683e-06` with N block `4.2577e-09`, stopped after
6 of 20 steps on `moment_extension_failure`, so "1.13 times more accurate"
(`1.9408 / 1.7231 = 1.126`) is right; the retry ladder is
`1.2277282102425304e-05`.

The motion-scale inertness claim reproduces exactly: `frontier_motion_scale`
100 gives `0.13133511462184333` against 1's `0.1313351146218195`.

### The affine-map claim, checked at probes of my own

Each of the three maps was rebuilt here in its own loop from the public
functions, then compared against those functions at twelve random probes per
map drawn at three scales (`0.05`, `0.5`, `2.0`) with a seed the module does
not use, and at five states taken from the saved trajectories:

| matrix | rows | affine in | maximum absolute reconstruction error, random probes | at trajectory states | build |
|---|---:|---|---:|---:|---:|
| retained joint Gram, `uncentered_joint_moment_matrix` | 8 | the 29 raw coordinates | `4.44e-16` | --- | `2.3 ms` |
| 24-row positive extension, `SymmetryReducedPositiveExtension.matrix` composed with `state_lower_moments` | 24 | the 44 state and 62 standardized frontier coordinates | `1.78e-15` | `1.11e-16` | `9.9 ms` |
| 62-row unified Gram, `CarriedWitnessGeometry.scaled_unified_matrix` | 62 | all 384 carried coordinates | `1.78e-15` | `5.55e-17` | `223 ms` |

**Claim confirmed.** The report's `1.1e-16` / `1.1e-16` / `2.8e-17` are the
module's own build-time probe at scale `0.05`; the error grows with the probe
magnitude, so at scale `2.0` it reads `1.8e-15`. Both figures say the same
thing, that the map is affine to machine precision, but the report should say
at what probe magnitude its number was taken.

The two structural facts of section 0 also reproduce: the 24-row design matrix
is `576 x 106` with rank 91, singular values `1.5726e-01` then `4.176e-16`, a
15-dimensional null space whose largest frontier component is `5.27e-16`, i.e.
the 15 unseen directions are propagated-state directions to machine precision.

### Corrections

**(1a) The repaired flow is 51 times worse than the unrepaired one on the
Eq. (22) internal energy, and the report does not say so.** Over `t = 0` to
`4`, offset-subtracted:

| carried-witness lane, `t = 0` to `4` | all-coordinate scalar RMS | Eq. (22) internal energy RMS | site-0 occupation RMS | N block |
|---|---:|---:|---:|---:|
| no repair | `0.4804854` | `0.0138792` | `0.0486426` | `0.7483502` |
| Dykstra 30 per step | `0.1843003` | `0.7092441` | `0.0069948` | `0.4993357` |
| Dykstra 200 per step | `0.1840868` | `0.7082989` | `0.0071565` | `0.4987658` |

The verdict "the repair buys physicality and nothing else ... the 2.6-fold
improvement comes from keeping the trajectory representable" is not supported
on the internal energy, where the repair is a 51-fold **regression**, and the
repaired flow at `0.709` is more than an order of magnitude worse on that
quantity than the projective event-guard route (`3.6201e-2`), the plain
archive EOM (`2.7587e-2`) and the unrepaired flow itself. The conventions
audit's correction 6 made the Eq. (22) internal energy a reported column for
exactly these lanes; it is in the artifact under
`dynamic_static_energy_rms_error` and is not printed.

**(1b) Two comparisons cross the raw and offset-subtracted conventions inside
one sentence.** The cheap route's `t = 0` offset is `8.08e-16`, so its own
figures are the same either way; the carried-witness figures it is compared
against are not.

- Section 1, `t = 0` to `0.05`: "Record lines on the same horizon:
  carried-witness strict always-on guard `3.236e-6`". That is the strict run's
  offset-subtracted figure; its raw figure on that horizon is `5.836875e-5`,
  and against the cheap route's `1.272e-5` the ordering reverses. The
  comparison is valid, in the offset-subtracted convention, and the column
  heading says "all-coordinate scalar RMS".
- Section 1, `t = 0` to `0.5`: "carried-witness no-guard anchor `3.19169e-5`".
  Offset-subtracted; raw is `6.734760e-5`. The factor against the cheap
  route's `2.0130e-3` is 63 in one convention and 30 in the other.
- Section 2 is unaffected: `0.18430` against the raw archive EOM `0.11089` is
  `1.662` in both conventions.

**(1c) "which is the recorded projective retry-ladder score" overstates a 3.6
percent agreement.** The control scores `1.2719373770656832e-05` and the
recorded ladder `1.2277282102425304e-05`; they differ by 3.6 percent, at
`0.141 s` against `112.46 s`. The cost statement carries the finding on its
own; the accuracy statement should say "within 3.6 percent of".

**(1d) Two tables have no artifact and one of them does not reproduce to its
printed digits.** Section 0's Dykstra iteration ladder and section 1's
six-time frozen-lift drift table are not in the provenance list of section 4,
and no script produces them. Reconstructing the ladder on the state the
repository test uses (`1e-3 * default_rng(0).normal(size=44)` added to the
prepared state, the perturbation of
`test_dykstra_reaches_the_cone_where_one_clip_does_not`) gives:

| policy | 24-row minimum after, here | as printed |
|---|---:|---:|
| before any repair | `-3.1447e-04` | `-3.145e-4` |
| `eigenvalue_clip` / `higham_nearest_psd` | `-2.1324e-04` | `-2.113e-4` |
| `dykstra` 3 | `-1.0005e-04` | `-9.850e-5` |
| `dykstra` 10 | `-7.4717e-06` | `-7.317e-6` |
| `dykstra` 30 | `-4.6906e-09` | `-4.552e-9` |
| `dykstra` 73 | `-1.528e-15` | `-2.9e-16` |
| `dykstra` 200 | `-6.33e-17`, stopping on its own after **76** iterations | "73, where Dykstra stops on its own" |

The decade structure is the report's, and the operating-point argument for 30
survives; the printed values are 1 to 3 percent away and the self-stopping
iteration count is 76, so the state the ladder was measured on is not the one
the report's description identifies. Name the state, or give the script.

**(1e) The peak-resident figures are not the artifacts'.** The report prints
`718 MB` and `751 MB`; the artifacts record `peak_resident_bytes` `804290560`
(767 MiB) for the control and `790364160` (754 MiB) for Dykstra 30. My rerun
recorded `804913152` in the artifact and `726 MB` at the `ram_guard`. Say
which clock the number comes from.

---

## 2. `projective-route-20260915b.md`

### (a) The exact start is an initial condition only — confirmed

`_exact_contracted_initialization`
(`paper_5/src/paper5/stability/apcm_projective_event_guard_analysis.py:579-687`)
propagates the exact truncated wavefunction once, at line 605, and returns
three things: the packed 44-coordinate state, the declared proximal log-det
completion at that state's own lower moments, and a record block. The caller
(`:1072-1110`) hands only `initial_state` and `initial_completion` to
`integrate_projected_apcm_ssprk3` (`:1158`). The exact wavefunction object is
not retained and `_propagate_exact_wavefunctions` is not called again. The
scoring reference is built by `_exact_archive_coordinates` at `:1240`, after
the rollout returns.

Inside the propagation modules there is exactly one exact-reference call,
`exact_holstein_joint_moment_initial_state` in
`prepare_projected_apcm_initial_state` (`projected_apcm.py:1528`), which is the
`t = 0` preparation and is not on the exact-start branch at all.
`apcm_projective_guard.py`, `apcm_moment_projection.py` and
`apcm_positive_extension.py` contain no exact-reference call.

**Confirmed: no exact information enters the propagation after the initial
condition.** One wording point: the fixed-dictionary predictor does carry the
exact K/P/D entrance correction and the exact active commutators, which are
closed-form operator identities and not exact-trajectory data; the report's
own configuration line says so, and the distinction is worth keeping explicit
when this claim is repeated.

### (b) The `t = 5.0` contracted completion — confirmed

Every value in the contraction table is in
`summary.json`'s `continuation.initial_exact_state`:

| quantity | artifact | as printed |
|---|---:|---:|
| declared completion certified with no retry | `initial_declared_completion_success true`, `initial_completion_retry_index -1` | "certified ... with no retry at all" |
| Newton iterations | 14, `certificate=interior Newton stationarity` | 14 |
| `reduced_gradient` | `2.623e-09` | `2.623e-09` |
| minimum scaled eigenvalue of the completed extension | `-6.177071025944706e-15` | `-6.177e-15` |
| box envelope ratio against the frozen box 1 | `0.24633203613687799` | `0.246332` |
| minimum eigenvalue of the retained joint moment matrix | `1.88252531801366e-05` | `1.882525e-05` |
| exact state norm | `1.0000000000000016` | same |

Naming: the result-first paragraph calls `2.623e-09` a "KKT backward error"
and the table calls it `reduced_gradient`, which is the name the completion's
own message uses. Use the one name.

### (c) The continuation — **the retraction count is wrong**

`t = 5.985` after 394 accepted steps, stopped by the `1,080 s` wall budget and
not by a refusal: confirmed
(`integration.message = wall_budget_exceeded after 394 of 1200 completed steps`,
`last_time 5.985`, autonomous wall `1,081.9 s`, peak `408977408` bytes). The
accuracy block is confirmed in every printed digit: all-coordinate scalar RMS
`0.0064864777576475185` over `t = 5.0` to `5.985`, offset `7.339e-15`,
blockwise rho / B / N / A / C `7.4733e-3` / `7.1183e-4` / `5.1005e-4` /
`1.4567e-3` / `8.9481e-3`, site-0 occupation `4.4477e-3`, static energy
`1.2183e-4`, carried lifts 611/571, stage-lift ladder 63/31, in-retraction
ladder 9/0, faces 2,363/1/0, guard stages 1,182/0/0, minimum scaled extension
eigenvalue `-2.32962e-11`.

**The stage-retraction count is not 2. It is 31.** Both counters agree:
`summary.json`'s `feasibility.total_stage_retractions` is `31`, and summing
`autonomous_trajectory.npz`'s `stage_retractions` gives `31` over 17 of the
394 steps, as does counting `stage_retraction_applied` true in
`outer_guard_stages.jsonl` (31 of 1,182 stages). The report's own ledger row
"the same ladder inside a stage retraction: certified / exhausted | 9 / 0"
already contradicts a total of 2.

The number 2 is the retraction count of the **last recorded step alone**
(`t = 5.985`). The progress lines the report's table is built from print the
per-step value, not a running total: `projected_apcm.py:1938` emits
`stage_retractions={stage_retractions[step + 1]}`, the count for that one
step. So the table's "zero stage retractions" at each sample point means
"no retraction on that particular step", not "no retraction so far".

What the 31 are, by time:

| interval | steps with a retraction | retractions |
|---|---:|---:|
| `t = 5.1325` to `5.1775` | 5 | 5 |
| `t = 5.935` to `5.985` | 12 | 26 (up to 3 on one step) |

This changes two readings in the report.

1. "The exact start passes all five walls without a single retraction" is
   false. Five retractions fall at `t = 5.1325`, `5.135`, `5.1425`, `5.1725`
   and `5.1775`, i.e. inside the `t = 5.085` to `5.235` interval that is the
   subject of the whole section.
2. "with 2 stage retractions instead of 122 to 269" should read "with 31".
   The comparison set, which I confirmed from the seven `t = 4` restart
   artifacts, is `t = 4.83` / 25, `5.085` / 122, `5.0975` / **3**, `5.1025` /
   216, `5.12` / 109, `5.1075` / 64 and `5.235` / 269 — so "122 to 269" is
   already a selection, and the carried-completion continuation that reached
   `5.0975` did so on 3 retractions, fewer than the exact start's 31.

The surviving claim, which the evidence does support, is narrower and should
be stated as it is: the exact start is the only run that passes `t = 5.235`,
and it runs 0.75 time units further than the furthest propagated-state
continuation, at 31 retractions against that run's 269. The retraction density
over its last 0.05 time units (26 in 20 steps, ending on the budget) is the
one number that argues the exact start was itself approaching a wall when the
clock stopped it, and the report should say so rather than claim the walls
were passed cleanly.

Naming: the ledger row "minimum unshifted retained joint-Gram eigenvalue over
the run | `1.882515e-05`" quotes `summary["feasibility"]["minimum_base_eigenvalue"]`,
which is the minimum over the trajectory of the smallest eigenvalue of
`uncentered_joint_moment_matrix` — the right quantity under the name every
other Paper V route uses. The key that literally carries that name in the same
block, `minimum_unshifted_retained_joint_gram_eigenvalue`, is
`electron_phonon_moment_matrix` and reads `7.531658e-05`. The clash is the
driver's (`apcm_projective_event_guard_analysis.py:1253` and `:1354-1357`), not
the report's, and it is a second instance of the audit's "two quantities, one
name" class.

### (d) The default path is bitwise unchanged — confirmed by a rerun

The report compares two stored artifacts. I reran the default rollout from the
**current worktree** — `--mode rollout --final-time 0.05 --time-step 0.0025`,
everything else defaulted — and compared to round 3's
`paper_v_apcm_projective_facegate_default_t005_h0025_20260915_v1`:

| array | `np.array_equal` |
|---|---|
| `times`, `states`, `approximate_archive_coordinates`, `base_minimum_eigenvalues`, `extension_minimum_eigenvalues`, `stage_retractions`, `accepted_substeps`, `outer_guard_triggered_stages` | all True |

All-coordinate scalar RMS `7.931830738563173e-05`, `50.57 s`. The stored
`defaultgate_t005_h0025_20260915b_v1` artifact is likewise bitwise equal to
round 3's on all eight arrays. **Gate confirmed on the code as it stands, not
only on the stored artifact.**

### Still outstanding from the conventions audit, and owned by this key

`_RECORDED_CWRMF_STRICT_ALL_COORDINATE_SCALAR_RMS = 3.24e-6`
(`apcm_projective_event_guard_analysis.py:100`) is still written into every
projective `summary.json` under the key
`cwrmf_strict_all_coordinate_scalar_rms_error_t0_to_0p05` (`:1365`). `3.24e-6`
is the carried witness's offset-subtracted figure; its raw figure over that
horizon is `5.836875e-5`. The worklog correction block records this as "to be
renamed" and it is not yet renamed — my own default-gate rerun wrote the wrong
key name again today.

### The (c) table, added after this review began, repeats the mis-read

The report was rewritten at 09:59 and committed as `59e3dc04`; section (c) is
now complete. Its table reproduces from
`output/local_runs/paper_v_apcm_projective_t4restart_ladder_h000125_t8_20260915_v1`
and `..._t4restart_ladder_facecontinuation_t8_...` except in two rows:

| row | as printed | artifacts |
|---|---|---|
| stage retractions, `h = 0.00125` | 6 | **64** (`total_stage_retractions` and the per-step sum agree; 6 is that run's last recorded step's own count, exactly as 2 was for the exact start) |
| stage retractions, `h = 0.0025` | 109 | 109, correct — so the table sets one run's last-step count against the other's run total |
| "minimum unshifted retained joint-Gram eigenvalue / minimum scaled positive-extension eigenvalue" | `7.605570e-04` / `2.411796e-06` against `3.0567e-03` / `2.262e-07` | the left column is `minimum_base_eigenvalue` and the right column is `minimum_unshifted_retained_joint_gram_eigenvalue`, two different matrices. On each key the two runs **agree**: base `7.605570e-04` and `7.605585e-04`; electron-phonon `3.0567290790e-03` and `3.0567290914e-03`. The row as printed shows a 4x difference where there is none. |

Everything else in the (c) table is the artifacts': `t = 5.1075` in 886 of
3,200 steps against `t = 5.12` in 448 of 1,600; budgets `1,020 s` and
`2,400 s`; walls `1,114.8 s` and `2,583.8 s`; stage-lift ladder 108/78 against
115/149; in-retraction ladder 19/12 (the `h = 0.0025` run's is 42/36, printed
as "not printed"); dynamic RMS `4.477829e-02` against `4.532038e-02`; raw
`4.085900e-02`; inherited `t = 4` offset `2.503676e-01`, identical in both;
faces 5,418/0/0; peak `516538368` bytes. The 800 samples up to `t = 5.0` of
reading 2 are confirmed; the "about 120 s" and "the other 995 s" split is not,
because `progress.jsonl` carries no timestamps and no other artifact records
a within-run timing.

At the corrected counts, two of the three readings change.

- Reading 1, "6 stage retractions against 109 ... the retraction storm is a
  step-size artifact", becomes 64 against 109. Per accepted step that is
  `0.072` against `0.243`, a 3.4-fold reduction in rate and a 1.7-fold
  reduction in count; both runs concentrate their retractions in 17 and 14
  steps respectively. A step-size effect, not a storm removed.
- Reading 3, "a rollout that arrives at `t = 5` with the exact state runs
  through the same interval at 2 retractions per 394 steps", becomes 31 per
  394 steps, `0.079` per step. That is not distinguishable from the
  step-halved propagated restart's `0.072` per step, so the sentence no longer
  separates the exact start from a refined propagated start. What does
  separate them, and is worth saying instead, is the comparison **at equal
  step**: at `h = 0.0025` the exact start retracts on `0.079` of its steps
  against the default `t = 4` restart's `0.281` (122 over 434) and the
  carried-lift-plus-face restart's `0.545` (269 over 494).

The commit message of `59e3dc04` also carries the wrong figure ("exact t=5
state runs to t=5.985 with 2 retractions").

---

## 3. `cwrmf-chtc-20260915b.md`

### The gates rerun here

| gate | result here |
|---|---|
| `test_accepted_working_set_can_be_restored_for_continuation` | passed, `0.07 s` |
| `test_default_checkpoint_carries_the_scalars_and_no_working_set` | passed, `0.37 s` |
| `test_resumed_rollout_under_carried_basis_reproduces_the_unbroken_one` | passed, `60.78 s` |
| `test_resume_below_the_exact_reference_zero_tolerance_still_scores` | passed, `0.68 s` |
| `test_declared_stop_request_halts_the_rollout_at_an_accepted_step` | passed, `0.06 s` |
| `test_wall_budget_is_off_by_default_and_a_budget_not_reached_is_inert` | passed, `0.67 s` |
| `test_a_reached_wall_budget_stops_cleanly_and_leaves_a_resume_point` | passed, `0.63 s` |
| all seven together | `7 passed in 64.24 s`, peak `1,262 MB` |
| `test_apcm_carried_witness.py` test count | 48 collected, as stated |
| bundle SHA-256 | `edffe887de74267dd557973df4532383524e559d9dba13d098bfea5844d5110f` |
| bundle bytes / entries | `520921` / 101 |
| `chtc/validate_condor_submit_lifecycle.py` on the eight submit files | exit status 0 |

The bundle's copies of the lane directory and of the validator are byte-identical
to the worktree's, so the second validator pass the report records is the same
pass.

The section (a) resume table reproduces from the artifacts under
`output/local_runs/paper_v_cwrmf_resume_20260915_v1/`, recomputed here as the
maximum absolute difference of the last recorded row against the unbroken
rollout:

| quantity at `t = 0.005` | basis carried | hint only |
|---|---:|---:|
| 31 reported archive coordinates | `0.0` | `0.0` |
| 384 carried coordinates | `0.0` | `2.026859454590325e-08` |
| completion correction norm | `0.0` | `1.361523080438965e-07` |
| readable-rate residual | `0.0` | `3.92248374790578e-08` |
| minimum shifted lower bound | `0.0` | `1.3644930807725822e-13` |
| minimum unshifted 62-row Gram eigenvalue | `0.0` | `1.3840175749853695e-13` |
| velocity margin | `0.0` | `1.7359977755315725e-07` |
| active critical modes | 35 (unbroken 35) | 36 |

and at `h = 0.0015` over `t = 0` to `0.003`, hint only: `9.317855447515429e-08`
on the carried coordinates, `6.246955203647178e-08` on the completion
correction norm, `1.538334993833601e-07` on the velocity margin, 33 against 32
critical modes. The "negative result first" block also reproduces exactly:
balanced always-on, `t = 0` to `0.05`, `carried_accepted_basis` with margin 2
costs `879.7751007919724 s` of Clarabel solve tier against the balanced
default's `257.4393259570061 s` (`3.417x`), maximum readable-rate residual
`2.8753297628555923e-03` against `2.9029520093262277e-05`, peak `3,876 MiB`.

The balanced smoke run reproduces: `2.646 s`, dynamic all-coordinate scalar
RMS `3.235916676961365e-06`, zero Clarabel solve-tier calls, 61 Gram
eigendecompositions, 40 unguarded velocity evaluations, peak `651 MiB`, and
the resumed step reaching `t = 0.0525` in `0.497 s` at `1.957433e-07`.

### What the inner runner can and cannot overwrite

`chtc/paper_v_cwrmf_event_guard_accel_20260915/run_cwrmf_event_guard_accel.sh`
contains **no deletion of any kind** — no `rm`, no `find -delete`, no
truncation — and every path it writes is inside the job's own sandbox:
`mkdir -p` on `raw_outputs/<batch>`, `logs/<batch>` and `archives_by_receipt`;
`tar -czf` of the run directory into the working directory; `mv` of that
archive into `archives_by_receipt/<archive sha256>/`; and the receipt written
beside it. The Apptainer wrapper adds nothing destructive and passes the four
named options plus `OMP_NUM_THREADS` through `--cleanenv`; the empty-string
defaults it exports are absorbed by the runner's `:-` substitutions, so an
unset option still reaches the documented default.

So **the runner cannot delete or overwrite a remote archive.** The overwrite
risk is one level up, in the stage-out, and it is real:

- `transfer_output_files = raw_outputs, logs, archives_by_receipt` with
  `preserve_relative_paths = True` writes into the submit directory on the AP,
  replacing same-named files. `archives_by_receipt/<sha>/` is content
  addressed, so the report's and the README's claim that two stage-outs cannot
  overwrite one another holds **for the archives**.
- `raw_outputs/<batch>/<run_id>/` is **not** content addressed, and it is
  where `checkpoint.npz` lives — the file segment `k+1` names in its
  `transfer_input_files`. Every submission carrying the same run id replaces
  it. The run id is hard-coded in each submit file's `arguments` line
  (`accel_s01_t0000_t0500`, ...), so resubmitting a `.sub` file unchanged
  reuses it.

**That is exactly the flow the wall budget was added to serve.** The README
says of a `wall_budget` return: "Submit that same segment's `final_time`
again, from the checkpoint it left, with a new `run_id`; the driver reads the
resume time from the checkpoint, so **nothing else in the submit file
changes**." Three things do have to change, and the shipped files support none
of them:

1. the run id in `arguments` (else the returned `raw_outputs/<batch>/<run_id>/`
   overwrites the very checkpoint being resumed from);
2. the fifth argument, `initial_state_file`, which in every shipped file names
   the **previous** segment's checkpoint, not the segment's own — s01's is the
   literal `none`, so a resubmitted s01 restarts from `t = 0` and its shorter
   checkpoint replaces the longer one, with nothing in the lane detecting the
   backwards move;
3. `transfer_input_files`, which must carry the segment's own checkpoint.

The `output` and `error` file names also embed the old run id. As written, a
`wall_budget` return requires hand-editing a submit file, and the README's
sentence says the opposite.

Two smaller points on the lane. The lifecycle validator checks bounded
factories and the explicit ordinary-held release mode; it does not look at
`transfer_input_files`, `transfer_output_files`, `when_to_transfer_output` or
the argument contract, so "lifecycle-clean" is a narrower gate than the
report's ledger row suggests. And `when_to_transfer_output = ON_EXIT` is
correctly justified: the driver does create `--output-directory` with
`exist_ok=False` (`apcm_carried_witness_analysis.py:366`).

### The wall-budget contract, checked at the lane's own step

The contract reads as described in the source: the deadline is the driver's
own start plus the budget (`:548-552`); the predicate is consulted only after
an accepted step and never on the last one
(`apcm_carried_witness.py:4185-4196`, guarded by `step + 1 < intervals`); and
`checkpoint(step + 1, ...)` runs at `:4163`, before the predicate at `:4186`,
so the checkpoint does record the step the rollout stops on.

The report's default-path inertness test runs over `t = 0` to `2e-8` at
`h = 1e-8`, two steps. I extended it to the lane's own configuration —
balanced, event-triggered, hard floor `2e-4`, `max_threads_8`, `t = 0` to
`0.05` at `h = 0.0025` — run once unbudgeted and once with
`--maximum-wall-seconds 600`:

| array | `np.array_equal` |
|---|---|
| `times`, `carried_states` (21 x 384), `approximate_archive_coordinates`, `minimum_unshifted_eigenvalues`, `minimum_shifted_lower_bounds`, `correction_iterations`, `readable_rate_residuals`, `completion_correction_norms`, `critical_modes` | True |
| `maximum_atom_seconds` | False, up to `1.50e-3 s` (4.3 percent); wall time, not a propagated quantity |
| `velocity_margins` | False only because the array is NaN after its first entry in **both** runs |

The unbudgeted summary carries neither `maximum_wall_seconds` nor
`stop_reason`; the budgeted one records `600.0` and `final_time`. **The
default path is inert at the lane's own step, not only at `h = 1e-8`.** Note
for the test author: `test_wall_budget_is_off_by_default_and_a_budget_not_reached_is_inert`
compares every array except `maximum_atom_seconds` with `np.array_equal`,
which a NaN entry fails; it passes today because `velocity_margins` happens to
be finite at `h = 1e-8`, and it would fail at the lane's own step for a reason
that has nothing to do with the wall budget.

### The bundle pins a working tree, not a commit

`paper_5/src` inside `edffe887...` is byte-identical to this shared worktree's
`paper_5/src` as it stood at 09:54, which means it carries five modules of two
other keys' **uncommitted** work — `cheap_projection_route.py`,
`cheap_projection_route_analysis.py`, `projected_apcm.py`,
`apcm_projective_guard.py` and `apcm_projective_event_guard_analysis.py` all
differ from `HEAD` (`4510bd01`) inside the bundle. The two modules the lane
actually executes, `apcm_carried_witness.py` and
`apcm_carried_witness_analysis.py`, match `HEAD` exactly, so the lane's own
behaviour is pinned and the digest is honest about the bytes it names; but the
digest corresponds to no commit and cannot be rebuilt from git, and a bundle
destined for a remote submit root should not carry three agents' in-flight
edits. Rebuild it from a commit, or state in the report that it is a
working-tree snapshot and name the five files.

### Smaller

- The resumed-segment wall seconds printed in section (a), `30.4 s` and
  `44.4 s`, are not in the artifacts: `autonomous_wall_seconds` reads
  `28.485163542005466` and `42.63702950000879`. Say which clock.
- "the dynamic all-coordinate scalar RMS `3.235917e-06` ... reproduces the
  strict always-on anchor's `3.23592e-06`" is true but is not a check on the
  guard: the smoke run's value is **bitwise** the no-guard flow's
  `3.235916676961365e-06`, and the conventions audit's correction 7 records
  that strict, balanced and no-guard agree to seven figures on that horizon.
  The report's own sentence that the run "is not evidence that the flags are
  fast" should extend to accuracy: it is not evidence about the guard either.

---

## The worklog blocks

`worklog:4595-4712` (CWRMF resume and the CHTC lane) and `worklog:4713-4775`
(cheap projection) carry the same figures as their reports and no figure that
the reports do not. Everything I checked in them reproduces, with two
exceptions already listed above: the resumed-segment `30.4 s` / `44.4 s`, and
the cheap-projection block's record row `-0.5173` for the raw archive EOM's
minimum unshifted retained joint-Gram eigenvalue, which is inherited from
`worklog:4080` and which the conventions audit's own archive-EOM review lane
did not re-derive.

The projective key has **not** written a worklog block: `59e3dc04` added the
report, the source and the tests only. The two wrong retraction counts have
therefore reached the report and the commit message but not the worklog, and
correcting them now keeps them out of it.

## Corrections the parent should apply

**`cheap-projection-20260915b.md`**

1. Add the Eq. (22) internal energy RMS to the carried-witness `t = 0` to `4`
   table: no repair `1.3879e-2`, Dykstra 30 `7.0924e-1`, Dykstra 200
   `7.0830e-1`, all offset-subtracted; and reverse the verdict's "buys
   physicality and nothing else" to say that the repair improves the
   coordinate RMS 2.6-fold and the site-0 occupation 7-fold while making the
   Eq. (22) internal energy 51 times worse.
2. State the convention on both carried-witness comparisons: `3.236e-6`
   (`t = 0` to `0.05`) and `3.19169e-5` (`t = 0` to `0.5`) are
   offset-subtracted figures whose raw counterparts are `5.836875e-5` and
   `6.734760e-5`; the cheap route's own offset is `8.08e-16`.
3. Change "which is the recorded projective retry-ladder score `1.2277e-5`" to
   "within 3.6 percent of".
4. Name the state the section-0 Dykstra ladder was measured on, or give the
   script; on the repository test's own perturbation the printed values differ
   by 1 to 3 percent and Dykstra self-stops at 76 iterations, not 73.
5. Say what clock the `718 MB` / `751 MB` peaks come from; the artifacts
   record 767 MiB and 754 MiB.
6. Say at what probe magnitude the `1.1e-16` / `1.1e-16` / `2.8e-17`
   reconstruction errors were taken; at probe scale 2 they read `1.8e-15`, and
   the affine claim holds either way.

**`projective-route-20260915b.md`**

7. Replace "2 stage retractions" with **31** at
   `projective-route-20260915b.md:30`, `:179`, `:191` and `:240` (and in the
   commit message of `59e3dc04`), and "6 stage retractions" with **64** at
   `:39`, `:213` and `:228`. Three counters agree on 31 for the exact start:
   the summary's `feasibility.total_stage_retractions`, the sum of
   `autonomous_trajectory.npz`'s per-step `stage_retractions` array over 17 of
   the 394 steps, and `stage_retraction_applied` true on 31 of the 1,182 lines
   of `outer_guard_stages.jsonl`; and two agree on 64 for the `h = 0.00125`
   run. The printed 2 and 6 are each that run's **last recorded step's own
   count**.
8. Withdraw "the exact start passes all five walls without a single
   retraction" (`:156`). Five retractions occur at `t = 5.1325`, `5.135`,
   `5.1425`, `5.1725` and `5.1775`, inside the `t = 5.085` to `5.235` interval
   under study, and 26 more over `t = 5.935` to `5.985`.
9. Relabel the progress table: the driver's `stage_retractions=` line
   (`projected_apcm.py:1938`) is that step's own count, not a running total.
   Replace the "stage retractions" column with the retraction times, or drop
   it and give the run total.
10. Correct "2 stage retractions instead of 122 to 269": the comparison set is
    3, 25, 64, 109, 122, 216 and 269 across the seven `t = 4` restarts, and the
    carried-completion continuation reached `t = 5.0975` on 3.
11. Use one name for `2.623e-09`: the completion's own, `reduced_gradient`, not
    "KKT backward error".
12. Fix the (c) table's retained-Gram row (`:219`): the left column is
    `minimum_base_eigenvalue` (`7.605570e-04` and `7.605585e-04`, the retained
    joint Gram) and the right column is
    `minimum_unshifted_retained_joint_gram_eigenvalue` (`3.0567290790e-03` and
    `3.0567290914e-03`, the electron-phonon moment matrix). The two runs agree
    on each key to six figures; the row as printed reads as a 4x difference.
    Then restate readings 1 and 3 at the corrected counts: 64 against 109 is
    `0.072` against `0.243` retractions per accepted step, and the exact start's
    `0.079` per step is not distinguishable from the step-halved restart's
    `0.072`. The comparison that does separate the exact start is at equal
    step: `0.079` against the default `t = 4` restart's `0.281` and the
    carried-lift-plus-face restart's `0.545`, all at `h = 0.0025`.
    Reading 2's "about 120 s" / "the other 995 s" split has no artifact:
    `progress.jsonl` carries no timestamps.
13. Rename `cwrmf_strict_all_coordinate_scalar_rms_error_t0_to_0p05`
    (`apcm_projective_event_guard_analysis.py:1365`) to say that `3.24e-6` is
    the offset-subtracted figure; the worklog has carried this correction as
    outstanding since the audit and every projective run still writes the wrong
    key name. While there, the same summary's
    `feasibility.minimum_unshifted_retained_joint_gram_eigenvalue` is
    `electron_phonon_moment_matrix` while
    `feasibility.minimum_base_eigenvalue` is the retained joint Gram
    (`uncentered_joint_moment_matrix`) every other route means by that name.

**`cwrmf-chtc-20260915b.md`**

14. Fix the `wall_budget` resume instruction, in the README and in the report's
    `stop_reason` table. "Nothing else in the submit file changes" is wrong:
    the run id in `arguments`, the fifth argument `initial_state_file` and
    `transfer_input_files` must all change, because every shipped submit file
    names the **previous** segment's checkpoint and s01 names `none`. Either
    ship a resume submit file per segment, or make the run id and the
    checkpoint path submit variables, or state plainly that a `wall_budget`
    return requires editing three lines.
15. Qualify the overwrite claim. It is true of `archives_by_receipt/<sha>/`,
    which is content addressed. It is false of `raw_outputs/<batch>/<run_id>/`,
    which holds `checkpoint.npz` and is replaced by any stage-out reusing the
    run id — the same-run-id case the README's "even when a run id is reused"
    sentence explicitly invites.
16. Say that the bundle is a working-tree snapshot, not a commit, and name the
    five modules inside it that differ from `HEAD` `4510bd01`; or rebuild the
    bundle from a commit. The lane's own two modules do match `HEAD`.
17. Narrow the "lifecycle-clean" ledger row: `validate_condor_submit_lifecycle.py`
    checks bounded factories and the ordinary-held release mode, not the
    transfer contract or the argument contract.
18. Give the `t = 0` to `0.05` default-path inertness result at the lane's own
    step alongside the `h = 1e-8` test: with `--maximum-wall-seconds 600` every
    propagated array is bitwise the unbudgeted run's at `h = 0.0025`, and only
    `maximum_atom_seconds` moves. Note in the test that its `np.array_equal`
    sweep would fail on `velocity_margins`, which is NaN at any configuration
    where the guard engages.
19. Say which clock the resumed-segment `30.4 s` / `44.4 s` come from; the
    artifacts record `28.49 s` and `42.64 s`.
20. Drop the smoke run's RMS as a reproduction of the strict anchor: the value
    is bitwise the no-guard flow's, and on that horizon strict, balanced and
    no-guard agree to seven figures.
