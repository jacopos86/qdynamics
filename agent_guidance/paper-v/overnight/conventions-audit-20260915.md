# Accuracy conventions across the overnight record: audit and one-convention table

Reviewer pass over the worklog blocks dated 2026-09-14 and 2026-09-15
(`paper_5/notes/electron_phonon_closure_worklog_20260803.md:3873-4524`, which
gained the joint-Gram repair note at `:4500-4524` while this audit was being
written), the
eleven reports and eleven reviews under `agent_guidance/paper-v/overnight/`,
and section 0 of `agent_guidance/paper-v/paper-v-state-and-plan-20260914.md`.
Every number below was loaded from the saved arrays and recomputed here, not
read back from a `summary.json` or from the worklog. Recomputation scripts and
their output are under
`output/local_runs/paper_v_review_conventions_20260915/`
(`recompute.py`, `one_convention_table.py`, `archive_eom_lane.py`,
`one_convention_table.json`, `archive_eom_t4.json`). One run was performed:
the no-guard carried-witness flow over `t = 0` to 4, section (c); it ran on the
working tree while other agents were editing `paper_5/src`, and it reproduced
the record's four printed columns, which is the only parity statement made
here about that tree.

## (a) Every accuracy convention in the record, and its defining function

`d = approximate - exact` on the 31 closed scalar coordinates;
`dd = d - d[0]`; `l2 = ||d||` per sample. Blocks are
rho 0:3, B 3:7, N 7:11, A 11:17, C 17:31.

| name as written in the record | what it computes | defining function, file:line | reported by |
|---|---|---|---|
| all-coordinate scalar RMS | `sqrt(mean(d**2))` over samples and all 31 coordinates | `_accuracy_metrics`, key `all_coordinate_scalar_rms_error`, `paper_5/src/paper5/stability/apcm_carried_witness_analysis.py:307` | projective route (every `t = 0` to 4, `t = 0` to 0.5 and `t = 0` to 0.05 number); carried-witness route **in the round-1 tier table and in every round-2/3 report** |
| dynamic all-coordinate scalar RMS | `sqrt(mean(dd**2))`, the run's own `t = 0` offset removed | same function, key `dynamic_all_coordinate_scalar_rms_error`, `:310` | carried-witness route **in the parity-anchor block, the resume block and the plan**; projective route at the `t = 4` restarts only |
| blockwise scalar RMS / blockwise dynamic scalar RMS | `sqrt(mean(d[:, block]**2))` and the same on `dd` | same function, keys `blockwise[<block>]["scalar_rms_error"]` `:291` and `["dynamic_scalar_rms_error"]` `:294` | projective route (rho/B/N/A/C lists); carried-witness C-block lines |
| site-0 occupation RMS | `sqrt(mean(err**2))` of `electron_density[0, 0]`, **not doubled** | same function, keys `site_0_occupation_rms_error` `:321`, dynamic `:324`; occupation built at `:262-273` | carried-witness and projective routes |
| interaction-energy RMS (2026-08-08 wording) = total-energy RMS (2026-08-07 wording) = static-energy RMS (driver wording) | `sqrt(mean(err**2))` of the Eq. (22) internal energy | same function, keys `static_energy_rms_error` `:327`, `dynamic_static_energy_rms_error` `:330`; `matrix_total_energy`, `paper_5/src/paper5/stability/matrix_reference.py:757` | carried-witness route; projective `t = 4` run |
| `coordinate_scalar_rms` | `sqrt(mean(d**2))` — numerically the same quantity as row 1 | **no repository function**; defined in an artifact-local script, `output/local_runs/paper_v_joint_gram_barrier_t20_20260915_v1/score_joint_gram_lane.py:97`, and documented there as the convention of `adaptive_positive_moment_analysis._score`'s `all_coordinate_scalar_rms_error`, `paper_5/src/paper5/stability/adaptive_positive_moment_analysis.py:110` | round-3 joint-Gram block only; the round-2 matched tables |
| complete 31-coordinate l2 time-RMS distance / `coordinate_l2_time_rms` | `sqrt(trapezoid(l2**2, t) / duration)` — trapezoidal in time | `time_rms`, `paper_5/src/paper5/stability/cone_long_horizon.py:249`, used at `:328`; copied from `_time_rms`, `pipelines/open_dynamics/score_coupling_controller_exact_reference.py:59` | round-2 boson-barrier block; round-3 joint-Gram block; plan section 3.2 |
| `time_rms_coordinate_l2_error` (and `coordinate_l2_sample_rms`) | `sqrt(mean(l2**2))` — an unweighted **sample** mean, not trapezoidal, despite "time_rms" in the name | `apcm_carried_witness_analysis.py:313`; second name at `score_joint_gram_lane.py:99` | not quoted in any 2026-09-14/15 worklog block; present in every projective and carried-witness `summary.json` |
| all-31 scaled RMS / `all31_scaled_rms` | `sqrt(mean((d / scales)**2))` with `scales` = per-coordinate `max(floor, max|path|, peak-to-peak)` of **the scoring run's own exact path** | `cone_long_horizon.py:346`; scales from `development_coordinate_scales`, `paper_5/src/paper5/stability/multi_coherent_scores.py:179-210` | round-2 boson-barrier block; round-3 joint-Gram block; the adaptive-packet line `0.063649` over `t = 0` to 98 |
| the six observable RMS values | `sqrt(mean(err**2))` per observable, over `site_0_occupation`, `site_1_occupation`, `electronic_energy`, `phonon_energy`, `electron_phonon_energy`, `internal_energy`; occupations are `2 * rho[i, i]` | `sample_rms`, `cone_long_horizon.py:263`, used at `:331`; observables at `closed_observable_trajectories` `:199`, names at `:55` | round-2 boson-barrier block (five of the six; `site_1_occupation` is not printed) |
| observable time-RMS | the same six under `time_rms` (trapezoidal) | `cone_long_horizon.py:330` | stored in every long-horizon `summary.json`; not quoted in the worklog |
| Hellinger distance / electronic-band weight of the polarization spectrum | Hann-windowed spectrum of `P = -Delta n / 2`, mean subtracted, sample step 0.2 | `pipelines/open_dynamics/analyze_archive_observable_spectra.py` via the long-horizon driver | round-2 boson-barrier block; plan section 3.2 |

Two structural facts that make the table necessary.

1. **Row 1 and row 2 differ by the run's `t = 0` offset, and only the
   carried-witness route has one.** The carried-witness 384-coordinate
   embedding of the canonical cutoff-16 correlated preparation reads back 31
   archive coordinates that differ from the exact reference at `t = 0` by
   `3.2442e-4` in Euclidean norm, of which `3.2025e-4` is in the A block,
   `4.538e-5` in C and `2.517e-5` in N; rho and B agree to `5e-16`. The
   projective route's offset is `8.08e-16`, so for it rows 1 and 2 are the
   same number. Every carried-witness accuracy figure therefore moves by a
   factor of about 18 at `t = 0` to 0.05 depending on which row is used
   (`5.836875e-5` against `3.235917e-6`), while no projective figure moves at
   all.
2. **The same words name two quantities that differ by a factor 2.**
   "site-0 occupation RMS" is `rho[0,0]` in `_accuracy_metrics:262-273` and
   `2 * rho[0,0]` in `closed_observable_trajectories:239` and in the
   adaptive-packet line's producer (`_observables`,
   `pipelines/open_dynamics/plot_paper_v_results_progression.py:308`). No
   2026-09-14/15 block crosses the two families, but the plan's section 3.6
   places the packet line beside the moment routes generally.

## (b) Where the record compares two conventions as one, or gives a number without its horizon

Worklog line ranges are of the 2026-09-14 and 2026-09-15 blocks.

| # | place | what is written | what is measured |
|---|---|---|---|
| 1 | worklog:4025-4028, round 1, projective block | "all-coordinate scalar RMS ... `7.9318e-5` ... against the carried-witness strict `3.236e-6` on the same horizon" | `7.9318e-5` is the raw row-1 quantity and `3.236e-6` is the carried witness's row-2 quantity. The comparison is valid in row 2 (the projective route's row 1 and row 2 coincide) but is labelled in row 1, where the carried witness reads `5.836875e-5` and the ordering reverses. The factor "24.5" is a row-2 factor; the row-1 factor is `1.36`. |
| 2 | source constant, `apcm_projective_event_guard_analysis.py:100` and `:1365` (working tree, 2026-09-15; other agents are editing this file, so re-locate by symbol) | `_RECORDED_CWRMF_STRICT_ALL_COORDINATE_SCALAR_RMS = 3.24e-6`, written into every projective `summary.json` as `cwrmf_strict_all_coordinate_scalar_rms_error_t0_to_0p05` | `3.24e-6` is the carried witness's `dynamic_all_coordinate_scalar_rms_error`; its `all_coordinate_scalar_rms_error` over that horizon is `5.836875e-5`. The stored key name states the wrong convention. |
| 3 | worklog:3962, round 1, tier table | row "all-coordinate scalar RMS against the exact reference, same runs: `5.836875e-5` / `5.827262e-5`" | Two different runs at two different horizons on one line: `5.836875e-5` is no-guard over `t = 0` to 0.05, `5.827262e-5` is strict over `t = 0` to 0.01. Neither horizon is stated, and both are row 1 while the anchor block twenty lines earlier quotes the same runs in row 2. |
| 4 | worklog:4201-4205, round 2, projective block | "`3.5137e-5` at `h = 0.0025` ..., `1.2875e-5` at `h = 0.00125` ..., `1.4635e-5` at `h = 0.000625`" | Three different sample grids (21, 41, 81 samples) under one horizon. The metric is an unweighted mean over samples, so refining the step also changes the measure. On the common 21-sample grid the three read `3.513696e-5`, `1.300620e-5`, `1.488741e-5`; the non-monotone refinement survives, the printed digits do not. The horizon `t = 0` to 0.05 is not restated in that sentence. |
| 5 | worklog:4194-4195, round 2 | "With the exact K/P/D entrance correction off the error is `2.1281e-3` (C `3.17e-3`)" | No horizon. It is `t = 0` to 0.05, 21 samples (`paper_v_apcm_projective_ablation_no_kpd_t005_20260915_v1`). |
| 6 | worklog:4206-4208, round 2 | "A control that solves the declared completion from the origin ... gives `5.1100e-5` with 4 retractions" | No horizon. It is `t = 0` to 0.05, 21 samples. |
| 7 | worklog:4183-4189, round 2 | "the N-block error against the exact reference is `2.250e-8` at `t = 0.0175`, steps to `2.282e-4` at `t = 0.02`" alongside "the N block from `2.17e-4`" | Two quantities, one name. The first is the instantaneous N-block Euclidean norm (`2.249966e-8`, `2.282038e-4`); the second is the N-block RMS over `t = 0` to 0.05 (`2.1728e-4`). |
| 8 | worklog:4245, round 2, boson barrier | "all-31 scaled RMS (adaptive-packet convention; the packet route's `t = 0` to 98 line is `0.063649`)" with this run's `0.700816` and `0.713667` | `all31_scaled_rms` divides by scales frozen from the scoring run's own exact path. This run's scales come from `t = 0` to 100 and the packet line's from its own `t = 0` to 98 development path, so the three numbers are not in one convention. Round 3 measured the size of this effect on the same trajectory: `0.602005` against `0.748812` for one Eq. (112) lane scored with two runs' scales (`joint-gram-t20-20260915.md:106-113`). |
| 9 | worklog:4246, round 2, boson barrier | the five observable RMS values against "(packet line: `0.013600` / ...)" | The packet line is over `t = 0` to 98 and this row is over `t = 0` to 79.95; the horizon appears in the row above, not in this one. The sixth observable, `site_1_occupation`, exists in the artifact (`0.33041054`) and is not printed. |
| 10 | worklog:4255-4261, round 2, boson barrier | "Definitional finding: ... (Hellinger `0.63807`, band weight `0.52250`, exact `0.05691` ...)" | These are the `10 <= t <= 100` window of `plain_archive_lane.json`, printed immediately under a table headed `10 <= t <= 79.8`, with no window on the sentence. |
| 11 | worklog:4313-4316, round 2, event-guard resume | "about 4.1 h for the guarded `0.38` time units against the record's `12.1 h` for the strict always-on `0.5`, i.e. roughly `2.9x` on the guarded part" | `43,692 s / 14,722 s = 2.97`, an unnormalised ratio of two different horizons **on two different machines**. Per time unit it is `2.25x`. On this machine the strict always-on anchor costs `1,471.11 s` for 20 steps = `73.6 s` per step over `t = 0` to 0.05, and the balanced event-triggered resume costs `14,722.32 s` for 153 steps = `96.2 s` per step over `t = 0.1175` to 0.5. No same-machine same-horizon strict run exists, so the guarded-part factor is unmeasured. |
| 12 | worklog:4453-4456, round 3, projective | "the best horizon is `t = 5.235` ... dynamic RMS `4.9241e-2` over `t = 4` to 5.235" set beside the committed path's `4.3845e-2` over `t = 4` to 5.085 | Two horizons. The matched number exists in the report and was dropped: over `t = 4` to 5.085 the carried-lift-plus-face-continuation run reads `4.337435e-2` (`projective-face-20260915.md:491`), i.e. **more** accurate than the committed path, not less. |
| 13 | worklog:4418-4421 and 4453-4456, and the report tables | the `t = 4` restart runs are quoted as "dynamic", the `t = 0` to 4 run as "all-coordinate scalar RMS" | Within one route the word "dynamic" means two different subtractions: at `t = 0` it removes an embedding offset of `8.08e-16` (a no-op); at the `t = 4` restart it removes the parent trajectory's accumulated error at `t = 4`, Euclidean norm `2.504e-1`. Raw and dynamic differ there by 7 to 19 percent (`4.073972e-2` against `4.384520e-2`; `4.143364e-2` against `4.924052e-2`). |
| 14 | worklog:4481-4483, round 3, joint Gram | "Over `t = 0` to 0.30 the lane's complete 31-coordinate l2 time-RMS distance from exact is `0.06420598` (`coordinate_scalar_rms` `0.01184757`, all-31 scaled RMS `0.10810603`)" | No comparison lane at that horizon in the block itself. **Answered while this audit was written**: the repair note appended at worklog:4500-4524 gives the lane's own base velocity, the plain archive EOM, as `0.07152712` over the same seven samples, ratio `0.898`. The round-3 block at worklog:4481-4483 still prints the three bare numbers; it should point at the repair note. |
| 15 | plan section 0, row 1 | "with the carried lift and the face carried across stages the `t = 0` to `0.05` RMS is `1.2277e-5` (N block `3.1e-8`) ... `14 s` to `t = 0.05` with the carried lift" | `1.2277e-5` with N block `3.1027e-8` is the **completion retry ladder** run, whose stored `stage_policy` reads `carried_completion: false` and whose `projection_settings.face_policy` reads `refuse` (`paper_v_apcm_projective_ladder_gate_t005_h0025_20260915_v1/summary.json`); it cost `112.46 s`. The carried-lift-plus-face configuration reads `1.287512e-5` with N block `1.8296e-5` at `h = 0.00125` and cost `14.18 s`. The row pairs one configuration's accuracy with another's cost. No run combines the two at `t = 0` to 0.05. |
| 16 | plan section 0, conic row | "`max_threads_8` `1.51x` under balanced only; ... carrying the accepted basis is `3.4x` slower" | `1.51x` was measured on the **strict** `t = 0` to 0.01 rollout (`92.418 s` against `139.883 s`); "balanced" describes its acceptance agreement (29/29 balanced, fails strict on 1 of 7), not the speedup. "Carrying the accepted basis" is `3.42x` slower in the balanced profile over `t = 0` to 0.05 and `1.46x` **faster** in the strict profile over `t = 0` to 0.01 (`104.227 s` against `152.448 s` of Clarabel tier seconds), `2.04x` with `max_threads_8`. The row prints only the adverse half. |
| 17 | worklog:4092-4094, round 1, projective `t = 4` | "The driver's static-energy RMS (`0.036201`) is not the record's interaction-energy RMS column and is not compared to it here." | They are the same quantity. The record's `0.01388` at `t = 0` to 4 reproduces here as `dynamic_static_energy_rms_error` `1.3879e-2`, and the record's `1.06402e-7` at `t = 0` to 0.5 reproduces as the same key. The comparison was available and is made in section (c); the projective route is the worst of the four lanes on it. |
| 18 | worklog:4089, round 1 | "the stage retraction ... on 1,068 of the 5,559 stages" against the same run's `feasibility.total_stage_retractions` `1041` | Two counts under one name: `outer_guard_stages.jsonl` records `stage_retraction_applied` true on 1,068 of 5,559 stages; the integrator's per-step counter sums to 1,041. |

Naming, flagged as instructed: `coordinate_scalar_rms` (row 6 of table (a)) is
a name introduced in round 3 for a quantity the repository already calls
`all_coordinate_scalar_rms_error`; it exists only inside an artifact script.
`coordinate_l2_sample_rms` is a second name for
`time_rms_coordinate_l2_error`. "Interaction-energy RMS", "total-energy RMS"
and "static-energy RMS" are three names for the Eq. (22) internal energy.

## (c) Every route at `t = 0` to 4 and `t = 0` to 0.05 in one convention

Convention for both tables: **dynamic all-coordinate scalar RMS**, the 31
closed scalar coordinates, unweighted over samples and coordinates, each run's
own `t = 0` offset subtracted (`_accuracy_metrics`,
`apcm_carried_witness_analysis.py:310`). The raw column and the offset are
printed beside it because the two orderings differ. Blocks and observables are
the same function's keys. All rows are on the `h = 0.0025` sample grid,
strong coupling (`lambda = 1.5`, `gamma = 0.5`, drive amplitude 1), phonon
cutoff 16, exact-correlated preparation, scored against the exact cutoff-16
driven reference. The exact reference arrays of the projective `t = 4` run and
of the carried-witness `t = 4` rerun are bitwise equal, so the rows are
comparable sample by sample.

### `t = 0` to 4, 1,601 samples

| lane | dynamic all-coordinate scalar RMS | raw | `t = 0` offset l2 | N block (dyn) | C block (dyn) | site-0 occupation RMS (dyn) | Eq. (22) internal energy RMS (dyn) | wall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| projective fixed-dictionary APCM, event-triggered outer guard, default | `2.604861e-2` | `2.604861e-2` | `8.08e-16` | `1.5504e-2` | `2.0175e-2` | `4.6452e-2` | `3.6201e-2` | `3,229.8 s` |
| Eq. (112) archive EOM (`closed_residual_subtracted_rhs`), SSPRK(3,3) | `6.082336e-2` | `6.082336e-2` | `0` | `5.8833e-2` | `4.8158e-2` | `3.8218e-2` | `4.0334e-2` | `0.63 s` |
| plain archive EOM (`closed_scalar_rhs`), SSPRK(3,3) | `1.108902e-1` | `1.108902e-1` | `0` | `7.8637e-2` | `7.6598e-2` | `2.9613e-2` | `2.7587e-2` | `0.62 s` |
| carried-witness flow, no Gram guard, `h = 0.0025` | `4.804854e-1` | `4.804803e-1` | `3.2442e-4` | `7.4835e-1` | `4.7234e-1` | `4.8643e-2` | `1.3879e-2` | `22.03 s` |

The plain archive EOM row reproduces all four printed columns of the record's
"matched raw 31-coordinate archive EOM" at this horizon (`0.11090`, C
`0.07660`, site-0 occupation `0.02961`, interaction energy `0.02759`;
worklog:3620-3633) to the digits printed, which identifies that lane as
`closed_scalar_rhs` with no Eq. (112) initial-residual subtraction. It does the
same at `t = 0` to 0.5, where the record prints `2.13e-2` and this lane gives
`2.127769e-2`. The carried-witness no-guard row is a rerun performed for this
audit and reproduces the record's `0.48049` / `0.47234` / `0.04864` / `0.01388`
to the digits printed.

Readings this table supports, in one convention: the projective route is the
most accurate of the four on the coordinates and on C; it is the **least**
accurate of the four on the Eq. (22) internal energy (`3.6201e-2` against the
carried witness's `1.3879e-2` and the plain archive EOM's `2.7587e-2`) and it
is worse than the plain archive EOM on site-0 occupation. The Eq. (112)
archive EOM, which costs `0.63 s`, is more accurate on the coordinates than
the plain archive EOM the record compares against, and is not in the record at
this horizon.

### `t = 0` to 0.05, 21 samples unless stated

| lane | dynamic all-coordinate scalar RMS | raw | `t = 0` offset l2 | N block (dyn) | C block (dyn) | Eq. (22) internal energy RMS (dyn) | wall |
|---|---:|---:|---:|---:|---:|---:|---:|
| Eq. (112) archive EOM, SSPRK(3,3) | `5.456228e-7` | `5.456228e-7` | `0` | `4.1704e-12` | `8.1184e-7` | `1.5894e-10` | under `0.01 s` |
| carried-witness flow, strict always-on guard | `3.235917e-6` | `5.836875e-5` | `3.2442e-4` | `4.5705e-8` | `8.2309e-7` | `2.0661e-10` | `1,471.11 s` |
| carried-witness flow, no Gram guard | `3.235917e-6` | `5.836875e-5` | `3.2442e-4` | `4.5705e-8` | `8.2309e-7` | `2.0661e-10` | under `1 s` |
| carried-witness flow, balanced always-on guard | `3.235916e-6` | `5.836875e-5` | `3.2442e-4` | `4.5705e-8` | `8.2308e-7` | `2.0661e-10` | `259.58 s` |
| projective + completion retry ladder, `h = 0.0025` | `1.227728e-5` | `1.227728e-5` | `8.08e-16` | `3.1027e-8` | `1.7241e-5` | `5.4153e-11` | `112.46 s` |
| projective + carried completion, `h = 0.00125` (41 samples; `1.300620e-5` on the 21-sample grid) | `1.287512e-5` | `1.287512e-5` | `8.08e-16` | `1.8296e-5` | `1.5341e-5` | `2.1479e-5` | `14.18 s` |
| projective + carried completion, `h = 0.000625` (81 samples; `1.488741e-5` on the 21-sample grid) | `1.463455e-5` | `1.463455e-5` | `8.08e-16` | `1.8768e-5` | `1.8378e-5` | `2.2034e-5` | `10.12 s` |
| projective + carried completion, `h = 0.0025` | `3.513696e-5` | `3.513696e-5` | `8.08e-16` | `8.9775e-5` | `1.9864e-5` | `1.0538e-4` | `41.98 s` |
| projective, event-triggered outer guard, default | `7.931831e-5` | `7.931831e-5` | `8.08e-16` | `2.1728e-4` | `2.0152e-5` | `2.5751e-4` | `50.58 s` |
| plain archive EOM, SSPRK(3,3) | `2.150615e-3` | `2.150615e-3` | `0` | `1.0015e-4` | `3.1987e-3` | `5.1618e-10` | under `0.01 s` |

Three readings this table supports and the record does not contain.

1. **The strict always-on guard is not separable from the no-guard flow at
   this horizon.** Strict, balanced and no-guard agree to seven significant
   figures (`3.235917e-6`, `3.235916e-6`, `3.235917e-6`), at `1,471.11 s` strict and `259.58 s` balanced
   against under a second unguarded. The plan's step-1 gate, "the strict prefix
   reproduces `3.24e-6` over `t = 0` to 0.05", is met by the no-guard run, so
   it certifies nothing about the guard; the guard's `t = 0` to 0.05 evidence
   is the 62-row Gram minimum (`-3.89e-12` carried, against `-1.27089e-2`
   unguarded at `t = 0.5`), not this RMS.
2. **The completion retry ladder and the carried lift are not
   interchangeable.** Their headline numbers differ by 5 percent
   (`1.227728e-5` against `1.287512e-5`) and their N block and Eq. (22) energy
   differ by nearly three and six orders of magnitude (`3.1027e-8` against
   `1.8296e-5`; `5.4153e-11` against `2.1479e-5`). The ladder repairs the
   structure the round-2 attribution identified; the carried lift moves the
   headline without repairing it.
3. **The Eq. (112) archive EOM is more accurate than every moment route at
   this horizon**, by a factor 5.9 against the strict carried witness and 22.5
   against the projective ladder, at no cost. Whatever the `t = 0` to 0.05
   figures are evidence for, it is not accuracy relative to a closed-form
   31-coordinate lane.

### What could not be recomputed

| line in the record | why |
|---|---|
| carried-witness strict always-on over `t = 0` to 4 | never run; the recorded strict cost is `43,692 s` for `t = 0` to 0.5 on the previous machine and `1,471.11 s` for `t = 0` to 0.05 here |
| 60-coordinate fixed-dictionary APCM, cells A / B / D over `t = 0` to 4 (`0.012580` / `0.046475` / `0.0125814`, worklog:2608-2664) and both cones over `t = 0` to 20 (`0.128607`) | the 2026-08 artifacts were lost with the previous machine; the plan's section 2 records this. These are `adaptive_positive_moment_analysis._score`'s `all_coordinate_scalar_rms_error`, i.e. the raw row-1 convention, and that driver emits no offset-subtracted key, so they cannot be placed in the dynamic column even if rerun without also recording the offset |
| carried-witness event-triggered balanced guard over `t = 0` to 0.05 | the first attempt's directory holds only `checkpoint.npz` and `progress.jsonl` (killed by the memory guard at `t = 0.1175`); the resume covers `t = 0.1175` to 0.5 only, so no `t = 0` to 0.05 or `t = 0` to 4 row exists for that configuration |
| projective route over `t = 0` to 4 with the retry ladder, the carried lift, or the face policies | only the default policy was run to `t = 4`; the variants exist as `t = 4` restarts (`t = 4` to 5.085 to 5.235) and as `t = 0` to 0.05 or `t = 0` to 0.5 rollouts |
| the record's `0.11066` raw archive EOM at `t = 0` to 4, `dt = 0.01` (plan section 3.1) | artifact lost; at `h = 0.0025` the same lane gives `1.108902e-1` here, so the two are the same lane at two steps and should not be quoted as one number |

The plain and Eq. (112) archive-EOM rows are review lanes produced for this
audit by `archive_eom_lane.py` (SSPRK(3,3) at `h = 0.0025` from the exact
cutoff-16 reference row at `t = 0`, scored on the projective run's own sample
grid and exact reference). They are diagnostic, not promoted evidence.
