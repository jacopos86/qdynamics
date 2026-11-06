# The cheap repair route: results of 2026-09-14 to 2026-09-18

**What this document is.** An exploratory results summary of the Paper V route
work carried out between 2026-09-14 and 2026-09-18 on branch
`claude/paper-v-routes-20260914`. It is **not** manuscript prose and **not**
evidence promotion: no number here is adopted evidence, every rollout's
`summary.json` carries `classification = exploratory_local_not_promoted`, and
adoption, promotion and demotion of any of these rows remain the user's
decision. Defaults are unchanged throughout; every behaviour named below is a
declared option whose absence is the earlier form. The manuscript
(`MATH/paper_details/paper_V_high_u_gkba.tex`) is untouched by this work.

**Audience.** Someone who knows the physics and has not read the execution
record.

**Sources, cited throughout in short form.**

| short form | file |
|---|---|
| `cheap` | `agent_guidance/paper-v/overnight/cheap-projection-20260915b.md` |
| `proj` | `agent_guidance/paper-v/overnight/projective-route-20260915b.md` |
| `worklog` | `paper_5/notes/electron_phonon_closure_worklog_20260803.md` |
| `plan` | `agent_guidance/paper-v/paper-v-state-and-plan-20260914.md` |
| `review-routes` | `agent_guidance/paper-v/overnight/review-routes-20260915b.md` |
| `review-27key` | `agent_guidance/paper-v/overnight/review-27key-chain-20260916.md` |
| `conventions` | `agent_guidance/paper-v/overnight/conventions-audit-20260915.md` |

**Configuration common to every number below unless a row says otherwise.**
Strong coupling `lambda = 1.5`, `gamma = 0.5`, drive amplitude 1, phonon
cutoff 16, exact-correlated preparation, SSPRK(3,3) at `h = 0.0025`. Scores
are the **all-coordinate scalar RMS of the 31 closed scalar coordinates**
against the exact cutoff-16 driven reference, `sqrt(mean(d**2))` over samples
and coordinates on the `h = 0.0025` grid from `t = 0` (`_accuracy_metrics`,
`paper_5/src/paper5/stability/apcm_carried_witness_analysis.py:307`;
`conventions` §(a)). Every route in sections 2 to 6 has a `t = 0` coordinate
offset of `8.08e-16`, so its raw and offset-subtracted ("dynamic") figures
agree to every printed digit. Carried-witness figures, wherever they appear,
are the **offset-subtracted ("dynamic")** quantity and are labelled as such.

---

## 1. What was being solved, and why the pre-existing routes were inadequate

The object is a moment closure of the driven Hubbard--Holstein dimer that
propagates through the pulse and past it, keeps its moments representable (the
retained joint Gram and the positive extension both at their cones), and costs
less than the object it approximates. **The exact cutoff-16 driven reference
propagates to `t = 4` in `0.7 s`** on this machine; that is the scale every
moment route is measured against (`plan` §0).

The routes on record before this work do not meet it.

| recorded route | accuracy, all-coordinate scalar RMS on the 31 closed scalar coordinates | horizon | cost | source |
|---|---:|---|---|---|
| 60-coordinate fixed-dictionary APCM, full cone plus retained joint-Gram controller ("cell A") | `0.012580` | `t = 0` to `4` | `3,399.6 s` | `worklog:2608-2664` |
| the same with the controller off, positive `M4` ("cell D") | `0.0125814` | `t = 0` to `4` | `3,268.6 s` | `worklog:2608-2664` |
| the same with a zero-cumulant prior plus controller ("cell B") | `0.046475` | `t = 0` to `4` | `52.9 s` | `worklog:2608-2664` |
| 60-coordinate fixed-dictionary APCM, both cones | `0.128607` | `t = 0` to `20` | `5,832.34 s`, hidden retraction on 5,998 of 8,000 steps | `worklog:2781-2792` |
| projective fixed-dictionary APCM with the event-triggered outer guard | `0.026049` | `t = 0` to `4` | `3,229.8 s`, of which `2,706.8 s` is the stage retraction on 1,068 of 5,559 stages | `worklog` 2026-09-14 round 1 |
| the same configuration toward `t = 20` | --- | stops in `5 < t < 6` | `moment_extension_degeneracy` after the drive ends | `worklog` 2026-09-14 round 1 |
| carried-witness radial moment flow (CWRMF), strict always-on guard | dynamic all-coordinate scalar RMS `3.19169e-5` | `t = 0` to `0.5` | `43,692 s` (previous machine); `1,471 s` for `t = 0` to `0.05` here | `worklog:3495-3498`, 2026-09-14 anchors |
| CWRMF, balanced bounded-depth event-triggered guard, hard floor `2e-4` | dynamic all-coordinate scalar RMS `2.4544e-5` | `t = 0.1175` to `0.5` | `14,722 s` for `0.38` time units, `6.05 GB` peak | `worklog` 2026-09-15 round 2 |

Three facts settle why the carried-witness guard could not be made fast.
`98.1` percent of a strict `t = 0` to `0.01` rollout is the Clarabel solve
(`142.350 s` of `145.172 s`, 36 calls; `worklog` 2026-09-14 round 1), so the
Python-level accelerations cannot reach it; of 22 solver candidates replayed
on 36 recorded conic instances none solves one instance faster under the
strict acceptance (`worklog` 2026-09-15 round 2); and seeding the spectral
bundle does not survive measurement, leaving the guarded atom with no lever
below about a factor two on this machine (`worklog` 2026-09-15 round 3). That
route is CHTC-sized, not laptop-sized.

The projective route's own wall is the conic stage retraction, not the guard:
the event-triggered outer guard has never fired on its own in any rollout
(0 of 7,882 stages across one round), and one outer selector solve costs about
`75 s` and `3.9 GB` (`worklog` 2026-09-15 rounds 2 and 3).

So before 2026-09-15 the record had one moment closure that reached `t = 20`
with both cones held, at `5,832.34 s` and an accuracy of `0.128607`, and
nothing that reached `t = 4` at a cost within three orders of magnitude of the
exact reference's `0.7 s`.

---

## 2. The method as it now stands

Five declared pieces. Each is a setting of a committed class; none is a
default. The state is fixed-dictionary; no conic solve runs inside the
physicality repair.

### (a) The fixed-dictionary state and its 27 entrance keys

The propagated object is the fixed-dictionary APCM state of a
`SymmetryReducedPositiveExtension`
(`paper_5/src/paper5/stability/apcm_positive_extension.py:308`) whose Gram
rows are nine base half-words plus one per active key. The dictionary used
here is the 15 entrance keys plus the twelve moments of two spin operators and
one boson coordinate: **27 entrance keys, 56 propagated state coordinates, 36
extension rows, 70 frontier keys, 41 lower keys** (`worklog` 2026-09-15 task 3;
`proj` §"projective-route, task 3"). `INITIAL_PROMOTION_CANDIDATE_KEYS` is the
entire declared enlargement, so `15 -> 31` is the only declared step and the
two blocks inside it are its only sub-steps.

`prepare_projected_apcm_initial_state`
(`paper_5/src/paper5/stability/projected_apcm.py:1513`) fails closed at 27
keys: the declared completion refuses at the contracted preparation (40 Newton
iterations, relative face support 35/36). The first rung of the declared
completion retry ladder (`logdet_shift 1e-8`) certifies there in 12
iterations, and every 27-key rollout is prepared at that rung.

The velocity is `CheapProjectionRoute.unguarded_velocity`
(`paper_5/src/paper5/stability/cheap_projection_route.py:1824`), which is
`FixedDictionaryProjectedAPCM.targets(...).augmented_retained_velocity`
(`projected_apcm.py:488, 698`) with no containment, no stage retraction and no
outer guard. Time stepping is `integrate_cheap_projection_ssprk3`
(`cheap_projection_route.py:1842`), SSPRK(3,3) at `h = 0.0025`.

### (b) The declared proximal log-det completion, re-selected every 16 steps

The lift --- the 70 standardized frontier coordinates --- is not propagated.
It is re-solved every 16 accepted steps by `CheapProjectionRoute.reselect_lift`
and `reselect_lift_decision` (`cheap_projection_route.py:1775, 1660`),
warm-started on the carried frontier
(`CheapProjectionRoute.carried_completion`, `:1575`), through
`SymmetryReducedPositiveExtension.complete`
(`apcm_positive_extension.py:748`) at
`APCMExtensionSettings.conic_tolerance = 1e-9`. Between re-selections the lift
is carried unchanged.

This cadence is the whole gap between the frozen lift and the projective
route's accuracy: at 15 keys the frozen lift scores `0.131335` over `t = 0` to
`4` and `N = 16` scores `0.0268423`, the projective event guard's `0.026049`
at one thirty-sixth of its wall (`cheap` second heading; `worklog` 2026-09-15
second pass). On the 27-key chain the 500 solves over `t = 0` to `20` cost
`8.32 s` each and carry `0.803` of the wall.

### (c) The Dykstra repair onto the affine Gram sets and the two cones

**Enabling fact.** The retained joint Gram (8 rows, the 29 raw moment
coordinates), the positive extension (36 rows at 27 keys) and the 62-row
unified Gram are *exactly affine* in the propagated coordinates ---
reconstruction errors `1.1e-16`, `1.1e-16`, `2.8e-17` at the module's own
build-time probe scale `0.05`, and `4.4e-16`, `1.8e-15`, `1.8e-15` at the
reviewer's independent probes at scale `2.0` (`review-routes` §1). A
physicality repair is therefore eigendecompose, clip, and one least-squares map
back through a pseudo-inverse factored once per rollout; no conic solve.

`AffineGramCoordinates.repair` and `AffineGramCoordinates._dykstra`
(`cheap_projection_route.py:1166, 1315`) run the alternation with
`policy = "dykstra"`, `target = "both"`, `cadence = "stage"` (three repairs per
accepted step, at the SSPRK(3,3) stages), `dykstra_iterations = 120`. Settings
are `CheapProjectionSettings` (`:123`); the floors `eigenvalue_floor`,
`retained_eigenvalue_floor` and `repair_tolerance` are all `0`.

### (d) The Eq. (22) energy reference in its drive-power form

**Enabling fact.** Eq. (22) is an exactly linear functional of the 29 raw
moment coordinates (offset zero, reconstruction `2.2e-16`), carried by seven of
the 106 parameters --- `-2 t_hop` on the electronic Bloch x coordinate, `2 g`
on each coherent-phonon real part and on two raw cross coordinates, `omega_ph`
on each diagonal raw normal coordinate --- and vanishing on the anomalous, the
entrance and all 62 frontier coordinates (`cheap` fourth heading). It is
therefore also linear in the realified Gram features, and the hyperplane on
which it equals a declared reference joins the two cones and the affine Gram
set as a third declared set of the same alternation.

`InternalEnergyFunctional` (`cheap_projection_route.py:604`),
`AffineGramCoordinates.energy_feature_functional` (`:1084`), the
`energy_reference` argument of `repair` and the `hyperplane` argument of
`_dykstra`. Mode `energy_reference_mode = "drive_power"`: the reference starts
at the closure's own Eq. (22) energy of the prepared or resumed state and is
advanced with the SSPRK(3,3) weights by
`gradient . (v(t, x, lift) - v(0, x, lift))`, both velocities being
`unguarded_velocity` at the **propagated** state. `drive_difference(0) = 0`
exactly, so `v(0, x)` is the drive-free velocity of the same coordinates and
**no exact or classical quantity enters the propagation** (verified
independently, `review-27key` §1).

The alternative declared mode is `post_pulse` (off until a declared drive-end
time, then the energy the trajectory itself carried there). Over `t = 0` to
`20` the constraint is met to `1.78e-15` over 23,802 constrained repairs with 0
hyperplane refusals.

### (e) The anchor-free measured-candidate acceptance rule, margin `1e-3`

`LiftReselectionRule(acceptance_margin=1e-3, candidates="measured",
anchor_weight=0)` with no trust radius (`cheap_projection_route.py:408`;
decision record `LiftReselectionDecision`, `:505`). The declared completion is
proximal to the zero frontier, not to the carried lift --- its warm frontier is
a starting point, not a term of the objective --- so every rule acts on the
answer. Under this rule a re-selection's returned iterate is accepted when it
passes the **measured** feasibility test (scaled minimum eigenvalue above
`-1e-8`) and improves the completion's declared objective
(`CheapProjectionRoute.declared_objective`, `:1612`) by more than the margin;
an uncertified iterate is therefore admitted on the measurement, not on the
certificate. Decisions of 500 on the 27-key chain over `t = 0` to `20`: **484
accepted (101 of them uncertified), 16 `infeasible`, 0 declined by the margin,
0 capped, 117 solver refusals** (`proj` §"projective-route, task 11").

**Driver.** The 27-key chains run through the artifact-local
`run_energy_segment.py` / `run_chain.py` under
`output/local_runs/paper_v_projective_dictionary_energy_20260915e/`, importing
the committed API from a `head_src` snapshot. The repository driver with the
same flag set at 15 keys is
`paper5.stability.cheap_projection_route_analysis`.

---

## 3. Results

### 3.1 The five-horizon table

Scores are the all-coordinate scalar RMS of the 31 closed scalar coordinates
against the exact cutoff-16 driven reference, cumulative from `t = 0` to the
stated horizon, on one grid (1,601 / 3,201 / 4,801 / 6,401 / 8,001 samples). A
**band** is the spread of the run and its one-setting twins --- the twins are
`energy_tolerance 1e-3`, the doubled Dykstra count, and re-selection
`conic_tolerance 1e-10` --- and is a measured sensitivity, not a statistical
error bar; the member count is given because it differs by row.

| chain, and the configuration the number comes from | `t = 4` | `t = 8` | `t = 12` | `t = 16` | `t = 20` |
|---|---:|---:|---:|---:|---:|
| **27-key ledger row** (the best 27-key number at each horizon; its configuration is the row named beneath it) | **`9.290e-3`** | **`2.593e-2`** | **`3.797e-2`** | **`4.821e-2`** | **`5.604e-2`** |
| anchor-free measured rule, no radius, Dykstra 120, from `t = 0` (task 11) | `9.290e-3` (4 members, `9.29e-3` to `9.95e-3`, 7.1 %) | `3.304e-2` (4, `3.010e-2` to `3.304e-2`, 9.8 %) | `4.060e-2` (1 run) | `4.610e-2` (1 run) | `5.604e-2` (1 run) |
| anchor-free measured rule, no radius, Dykstra 30 (task 8) | `1.001e-2` (4, `9.96e-3` to `1.001e-2`, 0.5 %) | `3.181e-2` (4, `3.180e-2` to `3.224e-2`, 1.4 %) | --- | --- | --- |
| the same with the ball trust radius `0.1`, Dykstra 30 (task 9) | `1.043e-2` (4, `1.004e-2` to `1.043e-2`, 3.9 %) | `3.181e-2` (4, `3.143e-2` to `3.216e-2`, 2.3 %) | `3.797e-2` (4, `3.797e-2` to `3.906e-2`, 2.9 %) | `4.821e-2` (4, `4.589e-2` to `4.861e-2`, 5.9 %) | `6.353e-2` (4, `5.792e-2` to `6.353e-2`, 9.7 %) |
| the ball base with Dykstra 120 from `t = 8` (task 10b) | --- | --- | `3.834e-2` (1 run) | `4.594e-2` (1 run) | `5.784e-2` (1 run) |
| recorded 27-key `drive_power` chain, earlier re-selection form, Dykstra 30 (task 4) | `1.143e-2` (2, to `1.292e-2`, 13.1 %) | `2.593e-2` (2, to `2.709e-2`, 4.5 %) | `4.251e-2` (2, to `4.332e-2`, 1.9 %) | `6.377e-2` (2, to `6.720e-2`, 5.4 %) | `8.519e-2` (2, to `9.088e-2`, 6.7 %) |
| recorded 27-key `post_pulse` chain, same form (constraint off during the drive) | `1.631e-2` (2, to `1.972e-2`, 20.9 %) | `4.538e-2` (2, `4.159e-2` to `4.538e-2`, 9.1 %) | `6.076e-2` (2, `4.612e-2` to `6.076e-2`, 31.8 %) | `6.385e-2` (2, `5.386e-2` to `6.385e-2`, 18.5 %) | `6.991e-2` (2, to `7.564e-2`, 8.2 %) |
| the task-11 configuration with the Eq. (22) hyperplane disabled (task 13) | `1.206e-2` (2, to `1.330e-2`, 10.3 %) | `4.853e-2` (2, to `4.946e-2`, 1.9 %) | `1.143e-1` (1 run) | `2.148e-1` (1 run) | `3.564e-1` (1 run) |
| 15-key cheap-projection ledger row, ball rule at Dykstra 30 | `2.72e-2` (4, 0.1 %) | `3.85e-2` (4, 0.2 %) | `9.89e-2` (4, to `1.043e-1`, 5.5 %) | `1.32e-1` (4, to `1.58e-1`, 19.3 %) | `1.33e-1` (4, to `2.07e-1`, 56.2 %) |
| cell A: 60-coordinate positive-`M4`, full cone plus controller | `1.2580e-2` | --- | --- | --- | --- |
| the both-cones line: 60-coordinate positive-`M4`, both cones | --- | --- | --- | --- | `1.28607e-1` |
| plain archive EOM (`closed_scalar_rhs`), SSPRK(3,3) at `h = 0.0025`, rescored | `1.10890e-1` | `1.38174e-1` | --- | --- | `1.93196e-1` |

Readings, once each. The `t = 4` row `9.290e-3` is `0.7385` of cell A's
`1.2580e-2`, and every member of its four-member band is ahead of task 8's
whole band by `1.001` to `1.078`. The `t = 20` row `5.604e-2` is `0.436` of the
both-cones `1.28607e-1`, ahead of the whole ball band by `1.03` to `1.13` and
of the recorded `drive_power` band by `1.52` to `1.62`. At `t = 8` the 27-key
row remains the **recorded** chain's `2.593e-2`: the anchor-free rule that
gains 15 percent at `t = 8` at 15 keys loses 17 to 24 percent there at 27 keys
(`proj` §"projective-route, task 8"), and the deeper repair applied from `t = 0`
is behind every 27-key chain at `t = 8`, `10` and `12`.

### 3.2 Internal energy, physicality, and wall

Eq. (22) internal-energy RMS is the sample RMS of the time-independent energy
of Eq. (22) on the stated horizon (`matrix_total_energy`,
`paper_5/src/paper5/stability/matrix_reference.py:757`). The physicality minima
are the minimum **unshifted retained joint Gram** eigenvalue and the minimum
**scaled positive-extension** eigenvalue over the stated horizon.

| chain | Eq. (22) internal-energy RMS, `t = 4` / `8` / `20` | min unshifted retained joint Gram | min scaled positive extension | horizon of those minima | wall |
|---|---|---:|---:|---|---:|
| anchor-free measured rule, no radius, Dykstra 120 (task 11) | `4.033e-3` / `4.888e-3` / `5.336e-3` | `-4.460e-4` (at `t = 11.04`) | `-3.936e-5` (at `9.4975`) | `t <= 20` | `5,183.6 s` |
| the same rule at Dykstra 30 (task 8) | `3.923e-3` / `4.781e-3` / --- | `-1.301e-3` | `-1.404e-4` | `t <= 8` | `1,886.3 s` to `t = 8` |
| the same with the ball radius `0.1`, Dykstra 30 (task 9) | `4.414e-3` / `5.361e-3` / `5.856e-3` | `-2.377e-3` (at `10.08`) | `-2.202e-4` (at `9.545`) | `t <= 20` | `4,844.7 s` |
| the ball base with Dykstra 120 from `t = 8` (task 10b) | identical to the ball's (same pinned reference) | `-4.420e-4` (at `11.047`) | `-4.187e-5` (at `9.498`) | `(8, 20]` | `5,393.7 s` |
| recorded `drive_power` chain (task 4) | `4.597e-3` / `5.752e-3` / `6.345e-3` | `-2.463e-3` | `-2.040e-4` | `t <= 20` | `4,794.4 s` |
| recorded `post_pulse` chain | `7.089e-2` / `1.373e-1` / `1.648e-1` | `-3.124e-3` | `-3.429e-4` | `t <= 20` | `3,886.9 s` |
| task-11 configuration, hyperplane disabled (task 13) | `4.565e-2` / `1.440e-1` / `1.371e0` | `-1.549e-12` (at `16.9575`) | `-1.350e-9` (at `0.0375`) | `t <= 20` | `3,514.0 s` |
| 15-key ball base | `3.31e-2` / `3.82e-2` / `4.09e-2` | `-7.9e-3` | `-2.6e-4` | `t <= 20` | `762` to `847 s` |
| cell A | `7.963e-4` at `t = 4` | `2.889e-5` | extended `M4` `-8.76e-7` | `t <= 4` | `3,399.6 s` |
| the both-cones line | `9.396e-4` at `t = 20` | `1.36819e-5` | both cones held | `t <= 20` | `5,832.34 s` |
| plain archive EOM | `2.759e-2` at `t = 4` / `1.824e-2` at `t = 20` | `-0.5173` at `t = 4` | --- | --- | under `1 s` |

**Machine.** Every wall in this table except cell A's and the both-cones
line's was measured on the replacement M4 Pro (24 GB), one heavy process at a
time, each rollout wrapped in `pipelines/shell/ram_guard.py --limit-mb 4500`,
peaks `624` to `699 MiB`. Cell A's `3,399.6 s` and the both-cones line's
`5,832.34 s` are the **previous** machine, the latter with four bounded
Clarabel/faer threads; the wall ratio between the two machines is unmeasured
(`review-27key` correction 4).

Three findings from this table, once each. (i) **No chain with the Eq. (22)
hyperplane on holds a cone**: past `t = 8` the retained cone is violated at all
4,800 samples in every constrained chain on record. (ii) The task-11
configuration has the shallowest cones of any chain run from `t = 0` with the
constraint, `5.1` and `5.4` times shallower than the same rule at Dykstra 30.
(iii) The Dykstra count is the setting that moves both the cones past `t = 8`
(`30` / `60` / `120`: `-2.38e-3` / `-1.09e-3` / `-4.42e-4`) and the score past
`t = 12` (`6.353e-2` / `5.792e-2` / `5.784e-2` at `t = 20`, applied from
`t = 8`).

### 3.3 Site-0 occupation, in one convention

Stated separately because the record carries two conventions for this quantity
and a factor 2 separates them (`review-27key` correction 3). The figures below
are all **`2 rho[0,0]`**, the record's progression-scorer convention; the
chains' own drivers report `rho[0,0]`, half of these.

| chain | site-0 occupation RMS at `t = 20`, `2 rho[0,0]` |
|---|---:|
| anchor-free measured rule, Dykstra 120 (task 11) | `0.136` |
| recorded `post_pulse` chain | `0.125` |
| task-11 configuration, hyperplane disabled (task 13) | `0.160` |
| ball rule, Dykstra 30 (task 9) | `0.162` |
| the both-cones line | `0.205` |
| recorded `drive_power` chain | `0.225` |

---

## 4. What is settled negatively

Each item is one measured result with its number, its configuration and its
horizon.

1. **The anchor.** A proximal anchor at the carried lift pays accuracy and
   never buys it back: at 15 keys over `t = 0` to `4` / `8` the anchor-free
   measured rule scores `0.027191` / `0.038538` against anchor weight 1's
   `0.031566` / `0.075551` (the anchor alone costs 29 and 57 percent), and
   weaker weights recover the accuracy in proportion without ever improving on
   none (`cheap` §"Task 8").

2. **The trust radius at 27 keys.** Under the `ball` policy at radius `0.1` the
   cap binds only inside `(0, 0.64]` (3 to 5 times of 200), costs 4 percent at
   `t = 4` (`1.043e-2` against the anchor-free band `9.96e-3` to `1.001e-2`,
   disjoint) and moves no number past `t = 8`: the radius-free chain continued
   from the same `t = 8` state scores `3.852e-2` / `4.848e-2` / `6.291e-2` at
   `t = 12` / `16` / `20`, ratios `1.014` / `1.006` / `0.990` to the ball base
   and inside the ball rule's four-member band at every horizon (`proj`
   §"projective-route, task 9", §"projective-route, task 10").

3. **The certified-only rule.** Refusing every uncertified return outright
   gives the widest band on record --- the base and its three one-setting twins
   spread `1.7`, `43.7`, `90.4`, `135.3`, `131.5` percent at `t = 4`, `8`,
   `12`, `16`, `20` at 15 keys, with two twins refusing 202 and 159 consecutive
   re-selections and running as the frozen-lift route over their last 6 to 8
   time units (`cheap` §"Task 11").

4. **The dictionary enlargement.** More entrance keys improve the 27
   coordinates outside the phonon-occupation block and worsen the four inside
   it: at cadence `N = 32` over `t = 0` to `4` the outside-block figure goes
   `3.276e-2`, `2.633e-2`, `1.769e-2`, `1.720e-2` at 15, 19, 27, 31 keys while
   the phonon-occupation figure goes `3.086e-2`, `3.916e-2`, `5.574e-2`,
   `8.645e-2`; and across eleven measured dictionaries (24 to 45 extension
   rows) the rank of the extension's derivative map over the four raw phonon
   normal coordinates is **1** in every one (second singular value at most
   `3.2e-17`), so no declared key reaches the other three (`proj`
   §"projective-route, task 3", §"projective-route, dictionary probe").

5. **The energy reference's offset.** It sets the Eq. (22) internal-energy RMS
   to four digits and nothing else: the `post_pulse` reference sits `+1.345e-2`
   above the exact post-pulse energy `-3.571397982`, `2.40` times the
   `drive_power` reference's `+5.614e-3`, and produces a *smaller* bias of the
   same sign in the phonon-occupation block (`0.81` to `0.84` of the task-11
   base's), while the recorded chain's `+6.710e-3` offset carries the smallest
   bias of all (`proj` §"projective-route, task 12").

6. **Cones without the energy constraint.** Removing the hyperplane is the one
   change that reaches both cones over the whole horizon (minimum unshifted
   retained joint Gram `-1.549e-12` at `t = 16.9575`, minimum scaled positive
   extension `-1.350e-9` at `t = 0.0375`, both at the scale of the arithmetic)
   and it costs `6.36` times the all-coordinate score and `257` times the
   Eq. (22) internal-energy RMS at `t = 20`, returning `32.2` percent of the
   wall; holding the cones and holding the energy are not independent settings
   of this route at 27 keys (`proj` §"projective-route, task 13").

7. **The per-coordinate motion scales.** The realified design over the 126
   movable coordinates has full rank 126 at `target = "both"`, so the
   least-squares map back is unique and every positive per-coordinate weight
   returns the same correction; where a weight can act (hyperplane on) it
   raises the per-repair displacement of the relative-mode population from
   `1.503e-4` to `4.49e-4` over `(6, 8]`, `2.99` times the quantity it was
   meant to cut by `1.78`, and `state_motion_scale` is exactly inert at `0.1`,
   `1.0`, `10.0` (`cheap` §"The declared per-coordinate metric of the map
   back").

8. **The energy slab.** Replacing the hyperplane by a band
   `|E - reference| <= tau` at `tau = 3e-4` cuts the replayed per-repair
   displacement `4.64` times on the stored `(6, 8]` samples, and a `t = 0` to
   `8` rollout at that band keeps 3 to 4 percent of it (`9.315e-3` /
   `3.313e-2` at `t = 4` / `8` against the task-11 base's `9.290e-3` /
   `3.304e-2`, ratio `1.003` at both; the phonon-occupation block `6.527e-2`
   against `6.779e-2` over `(4, 8]`), because in a rollout the stored internal
   energy sits at the band in both mean absolute value and maximum at all 800
   samples --- the slab binds at every repair and the route meets an equality
   displaced by the band (`cheap` §"The Eq. (22) reference as a slab rather
   than a point"). The band round read that saturation as the stage increment
   having already opened more Eq. (22) energy than the band; it has not. The
   offset the repair sees at the three SSPRK(3,3) stages is `4.158e-16`,
   `3.653e-16` and `3.203e-16` in mean absolute value on the task-11 base over
   `(6, 8]`, maximum `1.776e-15` at all 2,400 stage repairs, so what the band
   separates is the energy the cone chase opens inside the alternation
   (section 5) and nothing the stage increment carries in (`cheap` §"Whether
   the repair's displacement ... is the closure's own Eq. (22) energy drift").

9. **The drift identity.** The repair's displacement of the relative-mode
   population is not the closure's own Eq. (22) energy drift, because there is
   none: the closure's drive-free velocity changes the Eq. (22) energy by
   `4.352e-17` per step, the drift against the declared `drive_power` reference
   is `4.091e-16` per step, and the energy the repair actually supplied over a
   step is `6.145e-16` against a measured displacement of `1.989e-4` --- a
   ratio of `3.2e11`, correlation `+0.118` (`cheap` §"Whether the repair's
   displacement of the relative-mode population is the closure's own Eq. (22)
   energy drift").

10. **The declared step.** Reducing it does not reduce the repair's per-step
    displacement of the relative-mode population: taking the whole declared
    step with the three stage repairs chained from each of the 800 stored
    samples of `(6, 8]`, the displacement is `1.989e-4`, `1.495e-4` and
    `1.616e-4` per step at `0.0025`, `0.00125` and `0.000625` on the task-11
    base and rises at every halving on the recorded `drive_power` chain
    (`3.827e-4`, `4.812e-4`, `7.531e-4`), because the retained joint Gram
    violation the chase answers does not scale with the step --- observed
    power `+0.0000` over both halvings on both chains, the minimum scaled
    eigenvalue reading `-1.468e-5` and `-2.652e-4` at the stored state with no
    increment added at all, so it is the accepted state's violation and not
    the stage increment's --- while the scaled positive extension's does
    (`+1.0628` and `+0.8802` over the first halving), leaving the per-step
    quantity as the residue of two opposing terms whose cancellation ratio
    drifts through `1` (`0.958`, `1.030`, `1.161` on the base). No step
    reaches the closure's own per-step displacement of the same coordinate
    over the same window, `8.422e-5`: the closest the family comes is `1.77`
    times it, at `0.00125` on the base, and the sequence turns back up below
    that step. A `t = 0` to `8` chain would cost `2,073.4 s`, `4,146.9 s` and
    `8,293.8 s` at the three steps, the wall taken proportional to the
    declared step count at the task-11 configuration's measured `0.647950 s`
    per step (`cheap` §"Where the cone chase's own correction lands ...").

11. **The retained cone as the target for this coordinate.** At
    `target = "retained"` the chase lands `0.9864` of its squared correction
    length on the four raw phonon normal coordinates --- `0.0136` on the four
    coherent ones and exactly `0.0000` on the other 21 raw, the 27 dictionary
    and the 70 frontier coordinates, at every stage on both chains --- and
    displaces the relative-mode population by `7.526e-8` per stage repair on
    the task-11 base over `(6, 8]` and `8.989e-7` on the recorded chain, where
    that correction is `1.60` times longer than the extension cone's and
    displaces `2,556` times less: the share of a correction landing on the
    block does not predict what it displaces, the direction of this one being
    fixed by the affine eight-row Gram of `(1, b, b^dagger, sigma)` (`cheap`,
    same heading).

---

## 5. The one open defect: the relative-mode phonon population over `(6, 8]`

**The defect.** On the task-11 base the four phonon-occupation coordinates
score `6.779e-2` over `(4, 8]` against the recorded `drive_power` chain's
`4.538e-2` (`1.494`), and all of it sits in the second half of the window:
`8.278e-2` against `3.357e-2` over `(6, 8]` (`2.466`), where over `(4, 6]` the
base is ahead (`4.836e-2` against `5.469e-2`). This is the only part of the
chart on which the standing configuration is behind the earlier form at a
horizon it otherwise leads.

**The causal chain, as measured.**

| step | measurement | source |
|---|---|---|
| It is one coordinate, not three biases | of the block's `6.779e-2` over `(4, 8]`, `6.766e-2` (`0.9980`) is the **relative-mode population** `(phonon[0,0]+phonon[1,1])/2 - Re phonon[0,1]` alone and `1.883e-3` (`0.0278`) the three directions the extension cannot see; over `(6, 8]`, `0.9977`. In the site basis its single error of `-0.1472` appears as `-0.0736`, `-0.0736`, `+0.0736`, which the per-coordinate table had been reading as three separate biases | `proj` §"projective-route, dictionary probe" |
| No dictionary reaches the other three | the extension's online operator algebra is the exact spin-symmetric relative-mode reduction, so the four raw phonon normal coordinates reach it only through `v^dagger N v` with a real `v = (1,-1)/sqrt(2)`; the coefficient of `Im N[0,1]` is identically zero, and a row carrying the other three would have to be a word in a second boson mode, which the `MomentKey` form `(spin_up, spin_down, x_power, p_power)` cannot express | `proj` §"projective-route, dictionary probe" |
| Not a closure defect | the flow is Eq. (14b), exact and closed in the phonon density and the correlation; its velocity at the exact state integrates over `(4, 8]` to the exact relative-mode population's own change `2.323e-2` with residual `5.010e-4`, in a window where the base's error moves by `1.902e-1` | `proj` §"projective-route, relative-mode flow" |
| Not the flow difference | over `(6, 8]` the integrated signed difference between the closure's velocity of this coordinate and the exact reference's is `-6.514e-2` (base) against `-6.126e-2` (recorded chain), `1.063`, on a window where the coordinate's error stands at `1.907e-1` against `7.513e-2`, `2.539` | `proj` §"projective-route, relative-mode flow" |
| Not the entering state | the base enters `t = 4` at `-4.711e-2` and the recorded chain at `-6.637e-2`: the chain that ends `2.539` times worse enters `1.41` times better | `proj` §"projective-route, relative-mode flow" |
| It is the repair | over `(6, 8]` the repair displaces the coordinate `-7.083e-2` on the base and `+7.018e-2` on the recorded chain --- same size, opposite sign; from `t = 0` to `8`, of the `1.693e-1` by which the two errors differ at `t = 8`, the repair accounts for `1.749e-1` (`1.033`) and the flow difference for `-0.032` of it | `proj` §"projective-route, relative-mode flow" |
| Which part of the repair | with the hyperplane in the alternation the retained joint Gram is the mover (`2.131e-4` per repair on the base, `3.842e-3` on the recorded chain) and `0.970` / `0.980` of that block's correction to the 56 propagated coordinates lands on this one direction; with the hyperplane off the retained block is inert on it (`8.311e-8` and `9.026e-7` per repair, `2,564` and `4,258` times smaller) and `both` *is* `extension` | `proj` §"projective-route, repair displacement" |
| Why the hyperplane can do this | the Eq. (22) gradient is `0.5` on `raw_normal_00` and `0.5` on `raw_normal_11` and zero on both off-diagonal coordinates --- the same row as the relative-mode population's own `0.5`, `0.5` --- so the exchange rate is one unit of coordinate per unit of suppressed energy (fitted slope `+1.009268` on the base, `+0.448470` on the recorded chain) | `proj` §"projective-route, repair displacement" |
| What the hyperplane is actually annulling | not a drift (item 9 of section 4): the cone chase itself opens `9.732e-4` of Eq. (22) energy per stage repair on the base, positive at every one of 2,400, displacing the coordinate `+1.082e-3`; the hyperplane's annulment costs `-1.107e-3`; the constrained net `2.332e-4` is a near-cancellation of two terms `4.6` to `4.8` times larger | `cheap` §"Whether the repair's displacement ... is the closure's own Eq. (22) energy drift" |
| Where the sign comes from | the three SSPRK(3,3) stage repairs at weights `1/6`, `2/3`, `1`: stage signed means `+9.233e-5`, `-1.983e-4`, `+2.826e-5` on the base, whose weighted combination reproduces the in-rollout per-step displacement at correlation `+0.9999998` and slope `+1.000041` | same |
| Which cone's chase displaces it | at the same stage inputs with the hyperplane out, the positive extension's chase displaces the coordinate `1.113e-3` per stage repair on the base and `2.298e-3` on the recorded chain, `1.03` times the both-cone value on both chains, while the retained cone's chase displaces nothing (section 4, item 11): the whole of the displacement is the extension chase's, and direction rather than block is what separates a correction that moves this coordinate from one that does not | `cheap` §"Where the cone chase's own correction lands ..." |
| Where the annulment is taken up | with the hyperplane in the alternation the declared repair's correction puts `0.0536` of its squared length on the four raw phonon normal coordinates on the base and `0.2198` on the recorded chain, against the twin's `0.7022` and `0.9336`, and `0.8091` and `0.6872` on the 27 active dictionary coordinates | same |

**Remedies already refused, with their numbers:** turning the hyperplane off
(section 4, item 6, and `+2.713e-1` of one-signed displacement over `(4, 8]` on
the window rerun), the per-coordinate metric (item 7), the slab (item 8), more
entrance keys (item 4), the declared step (item 10) and the retained cone as
the target (item 11).

**The repair-side levers are exhausted.** On the energy channel there is no
Eq. (22) offset for any of them to trade against --- the offset a repair sees
is at the arithmetic's floor at all 2,400 stage repairs of each chain (item 8)
and the closure's own flow carries the declared reference exactly (item 9) ---
so the declared per-coordinate metric of the map back, the reference as a slab,
the Dykstra count, `state_motion_scale`, `repair_tolerance` and
`energy_tolerance` each act on a quantity that is zero, and the declared
reference's own level, `+5.614e-3` above the exact post-pulse energy, is a
one-time cost at the first repair and nothing per step thereafter. The two
levers that are not on that channel, the declared step and the retained cone
the chase answers, are refused in items 10 and 11.

**The one structural option the record supports** is a declared key whose
positive-extension row carries the phonon normal moments, which acts on how far
below the extension floor a stage increment falls rather than on what is done
about it afterwards. It is a change to the key algebra rather than a larger
dictionary: the extension's derivative map has rank 1 over the four raw phonon
normal coordinates at every declared key set (item 4), and a row carrying the
other three would have to be a word in a second boson mode, which the
`MomentKey` form `(spin_up, spin_down, x_power, p_power)` cannot express. What
would size it before any rollout, at the same 2,400 stage inputs: the extension
twin's displacement and its depth below the extension floor under the enlarged
row, against the `1.113e-3` and `2.298e-3` above and the scaled positive
extension's `-2.245e-4` on the base and `-4.336e-4` on the recorded chain
before the stage repair (minima `-1.060e-3` and `-1.138e-3`, below zero at
`0.8075` and `0.9846` of the stage repairs), both cones being reached after the
repair at `0.0083` of the base's stage repairs and none of the recorded
chain's (`cheap`, same heading, "What to try next").

---

## 6. Reproduction

Everything below is run from the worktree
`/Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914`
with `PY=/Users/lordskyler/Holstein_test/.venv/bin/python` and
`DIR=output/local_runs/paper_v_projective_dictionary_energy_20260915e`. The
27-key chains import the committed API from the snapshot
`$DIR/head_src_e3374f6f`, which was checked byte-identical to
`paper_5/src/paper5` outside `__pycache__` at each round's HEAD.

### 6.1 The `t = 4` and `t = 20` numbers (task 11 configuration)

```bash
export PYTHONPATH=$DIR/head_src_e3374f6f
RULE=(--reselection-acceptance-margin 1e-3 --reselection-candidates measured --reselection-anchor-weight 0)

# first segment
$PY pipelines/shell/ram_guard.py --limit-mb 4500 -- \
  $PY $DIR/run_energy_segment.py \
  --dictionary entrance15_plus_two_spin_boson \
  --output-directory output/local_runs/paper_v_dictionary_energy_power_27key_n16_measured_dyk120_base_20260918m_seg1 \
  --final-time 20.0 --lift-reselection-interval 16 \
  --energy-reference-mode drive_power --initialization-ladder \
  --wall-budget-seconds 1000 --record-reselections \
  "${RULE[@]}" --dykstra-iterations 120

# the remaining segments, each resumed at the previous one's last re-selection boundary
$PY $DIR/run_chain.py \
  --prefix output/local_runs/paper_v_dictionary_energy_power_27key_n16_measured_dyk120_base_20260918m_seg \
  --first 2 --last 12 --final-time 20.0 \
  --energy-reference-mode drive_power --wall-budget-seconds 1000 \
  --record-reselections "${RULE[@]}" --dykstra-iterations 120 \
  --log-directory $DIR

# scoring, on the cached exact reference
$PY pipelines/shell/ram_guard.py --limit-mb 2500 -- \
  $PY $DIR/score_energy_chain.py <the segment directories, in order> \
  --horizons 1 2 4 6 8 10 12 16 20 --exact-cache $DIR/exact_t20.npz \
  --output $DIR/task11_base_score_t20.json
```

The whole queue, including the three one-setting twins to `t = 8`
(`--energy-tolerance 1e-3`; `--dykstra-iterations 240`;
`--reselection-conic-tolerance 1e-10`), is `$DIR/queue_task11.sh`; the smoke
gate is `$DIR/smoke_task11.sh` with `$DIR/check_task11_smoke.py`.

### 6.2 Artifact paths for each ledger number

| number | artifacts | score file |
|---|---|---|
| `t = 4` `9.290e-3`, `t = 20` `5.604e-2` (task 11) | `output/local_runs/paper_v_dictionary_energy_power_27key_n16_measured_dyk120_base_20260918m_seg{1..6}` (each with `reselections.json`); twins `..._measured_dyk120_{etol1e3,dykstra240,conic1e10}_20260918m_seg*` | `$DIR/task11_base_score_t20.json`, `$DIR/task11_{etol1e3,dykstra240,conic1e10}_score_t8.json` |
| `t = 8` `2.593e-2` (recorded `drive_power` chain) | `output/local_runs/paper_v_dictionary_energy_power_27key_n16_20260915e_seg{1..6}`; tightened member `..._tightresel_20260916f_seg{1..6}` | `$DIR/power_chain_score_all.json`, `$DIR/tightresel_chain_score_t20.json` |
| `t = 12` `3.797e-2`, `t = 16` `4.821e-2` (ball rule, Dykstra 30) | `output/local_runs/paper_v_dictionary_energy_power_27key_n16_ball_{base,etol1e3,dykstra60,conic1e10}_20260917k_seg{1,2}`, `..._ball_base_20260917k_seg{3,4,5}` | `$DIR/task9_ball_base_score_t20.json`, `$DIR/tabulate_task9_band.log` |
| task 8's `t = 4` and `t = 8` four-member band | `output/local_runs/paper_v_dictionary_energy_power_27key_n16_measured_{base,etol1e3,dykstra60,conic1e10}_20260916i_seg{1,2}` | `$DIR/task8m_*_score_t8.json` |
| task 10b, Dykstra 120 from `t = 8` | `output/local_runs/..._ball_base_dyk120_20260918l_seg{3,4,5,6}` (`seg1`, `seg2` are links to the ball base's) | `$DIR/task10_ball_base_dyk120_score_t20.json` |
| task 13, hyperplane disabled | `output/local_runs/paper_v_dictionary_energy_none_27key_n16_measured_dyk120_{base_seg1..4,conic1e10_seg1,conic1e10_seg2}_20260918o` | `$DIR/task13_base_score_t20.json`, `$DIR/task13_conic1e10_score_t8.json` |
| the recorded `post_pulse` chain | `output/local_runs/paper_v_dictionary_energy_postpulse_27key_n16_20260915e_seg{1..4}`; tightened `..._tightresel_20260916g_seg{0..4}` | `$DIR/postpulse_chain_score_all.json`, `$DIR/tightresel_postpulse_chain_score_t20.json` |
| the `(4, 8]` window reruns (task 12) | `output/local_runs/paper_v_dictionary_energy_window48_27key_n16_task12_{a_rule30_recorded,a_certified_t11,b_energy_none,b_postpulse}_20260918n_seg1` | `$DIR/tabulate_task12.log` |
| the relative-mode measurements | `$DIR/relative_mode_flow_{w48,w68,w04,task12_w48}_20260918.json`, `$DIR/repair_displacement_{base,recorded}_w68_20260918.json`, `$DIR/closure_energy_drift_{base,recorded}_w68_20260918.json`, `$DIR/cone_correction_share_{base,recorded}_w68_20260918.json`, `$DIR/cone_step_scaling_{base,recorded}_w68_20260918.json`, `$DIR/cone_chained_step_{base,recorded}_w68_20260918.json` | the matching `.log` files |
| the 15-key ledger row | `output/local_runs/paper_v_cheap_projection_task10_{ball,carried}_{base,etol1e3,dykstra60,conic1e10}_t20_20260916_v1` | `output/local_runs/paper_v_cheap_projection_task10_trust_t20_20260916_v1/` |

### 6.3 The 15-key rows through the repository driver

```bash
PYTHONPATH=paper_5/src $PY pipelines/shell/ram_guard.py --limit-mb 3000 -- \
  $PY -m paper5.stability.cheap_projection_route_analysis \
  --output-directory <dir> --final-time 20 --time-step 0.0025 \
  --projection-policy dykstra --projection-target both --projection-cadence stage \
  --dykstra-iterations 30 --lift-reselection-interval 16 \
  --energy-reference-mode drive_power \
  --reselection-acceptance-margin 1e-3 --reselection-candidates measured \
  --reselection-anchor-weight 0 --reselection-trust-radius 0.1 \
  --reselection-trust-policy ball --wall-budget-seconds 1140
```

### 6.4 The exact reference

The exact cutoff-16 driven reference is `exact_holstein_driven_trajectory`
(DOP853, `rtol 1e-9`, `atol 1e-11`, `max_step 0.01`) from the zero-field
cutoff-16 ground state, cached as `$DIR/exact_t20.npz` (8,001 samples) and
`$DIR/exact_t8.npz` (3,201). An independent regeneration reproduced it
**bitwise** (`review-27key` preamble). It is a reporting input only and never
an online controller input.

---

## 7. What a reader should not conclude

**On status.** None of this is adopted evidence. Every rollout writes
`classification = exploratory_local_not_promoted`; the plan's row for this
route says "adoption pending"; the `t = 20` number is one run with no band of
its own. Nothing here has been carried into the manuscript.

**On conventions.** Six traps, each raised by a review file.

1. **Two accuracy conventions.** "All-coordinate scalar RMS" and its
   offset-subtracted ("dynamic") form differ by the run's own `t = 0` offset.
   Only the carried-witness route has one: its 31-coordinate readout of the
   canonical preparation sits `3.2442e-4` from exact at `t = 0`, so its `t = 0`
   to `0.05` figure is `3.235917e-6` offset-subtracted and `5.836875e-5` raw
   --- a factor 18, and the ordering against other routes reverses between
   them. Every figure in sections 3 to 6 has offset `8.08e-16` and is the same
   either way (`conventions` §(a), fact 1; `review-routes` §1, correction 1b).

2. **Site-0 occupation crosses conventions by a factor 2.** `rho[0,0]` in
   `_accuracy_metrics` and `2 rho[0,0]` in the progression scorer. Section 3.3
   is in the doubled convention throughout; the chains' own `summary.json`
   figures are half of it. The record's both-cones `0.204690` and
   plain-archive `0.286170` are `2 rho[0,0]` **trapezoidal** time-RMS. Reading
   the recorded `drive_power` chain's `1.126e-1` beside `0.204690` --- as the
   report originally did --- reverses the ordering: doubled it is `0.225`,
   *worse* than both cones (`review-27key` correction 3).

3. **`all31_scaled_rms` is not comparable across runs**, because the scales are
   frozen from each scoring run's own exact path; the multi-coherent adaptive
   packet route's `t = 0` to `98` line `0.063649` is therefore not a line these
   rows sit beside (`conventions` §(b), item 8).

4. **One quantity, several names.** "Interaction-energy RMS",
   "total-energy RMS" and "static-energy RMS" are all the Eq. (22) internal
   energy; `coordinate_scalar_rms` is a script-local alias of
   `all_coordinate_scalar_rms_error`; `time_rms_coordinate_l2_error` is a
   *sample* mean despite its name, while `coordinate_l2_time_rms` is
   trapezoidal (`conventions` §(a) and its naming note).

5. **Two machines in one column.** Cell A's `3,399.6 s` and the both-cones
   line's `5,832.34 s` are the previous machine (the latter with four bounded
   Clarabel/faer threads); "at `0.82` of the both-cones wall" was arithmetic on
   two clocks and is withdrawn (`review-27key` correction 4).

6. **Counts read from a last-step counter.** Two stage-retraction figures in
   the projective record were the last recorded step's own count, not the run
   total (2 for 31, 6 for 64); do not read a per-run count from a per-step
   field (`review-routes` §2(c)).

**On the results themselves.**

- **A band is a measured one-setting sensitivity, not an error bar.** Any
  ordering closer than the band is unresolved. At 15 keys one declared solver
  setting moves the `t = 20` score by 26 percent under the earlier
  re-selection form, and the exact flow amplifies the same perturbation by at
  most `1.16` over `t = 0` to `20` --- the amplification is the closure's, not
  the physics' (`cheap` sixth heading).
- **Chaining under this rule is not bitwise.** A chained run agrees with a
  single rollout to `2.5e-14` through three re-selections and can flip a fourth
  at the feasibility test's edge (`infeasible` against `accepted`, a `0.89`
  move, `d31` `3.3e-4` two samples later), so every band includes the segment
  boundary as a perturbation of the twins' kind (`proj` §"projective-route,
  task 8").
- **Reaching `t = 20` is not being physical.** Every chain in section 3 with
  the Eq. (22) hyperplane on violates the retained cone at all 4,800 samples
  past `t = 8`. The only chain that holds both cones is the one with the
  hyperplane off, and it is the worst 27-key chain on record at every horizon
  from `t = 8`.
- **Cell A is a different construction.** 60 coordinates closed with
  `PositiveFourthMomentCompletion` at `logdet_shift 1e-5`; it is a comparison
  line at `t = 4`, not the same object measured differently (`worklog`
  2026-09-15 task 3).
- **"Best on record at every horizon it reached" was withdrawn.** For the
  recorded `drive_power` chain it holds at `t = 4`, `6`, `8` and `12`, ties at
  `t = 16`, and is second at `t = 20` to the route's own `post_pulse` variant
  (`review-27key` correction 1).
- **Short horizons do not discriminate.** Over `t = 0` to `0.05` the Eq. (112)
  archive EOM scores `5.456228e-7`, better than every moment route at no cost,
  and the strict, balanced and no-guard carried-witness flows agree to seven
  figures there; that horizon is evidence about the cone, not about accuracy
  (`conventions` §(c)).
- **The energy hold is not an accuracy statement.** Holding Eq. (22) on the
  declared reference to `1.78e-15` says the hyperplane was representable at
  every repair; the reference's own level sits `5.614e-3` above the exact
  post-pulse energy, and that offset --- not the hold --- is what the
  internal-energy RMS column measures.
