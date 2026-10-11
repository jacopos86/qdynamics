# The archive closure's divergence, the cheap repair, the guard and their hybrid: classical results to 2026-10-11

**What this file is.** A consolidated summary of the classical results of
2026-10-10 and 2026-10-11 on branch `claude/paper-v-routes-20260914`.

- **Status.** It is exploratory and not promoted. No number here is adopted
  evidence. Every receipt that records a classification says
  `exploratory_local_not_promoted`.
- **Decisions that stay with the user.** Adopting evidence, promoting it,
  demoting it, and inserting anything into the manuscript
  (`MATH/paper_details/paper_V_high_u_gkba.tex`, untouched by this work).
- **What changed.** No default changed, and no module under `paper_5/src`
  was edited.
- **Supersession.** For everything it covers, this file supersedes
  `MATH/paper_facing/paper_V_high_u_gkba/cheap_repair_route_results_20260918.md`.
  That file is left unedited as the record of 2026-09-14 to 2026-09-18.
- **Sources.** No new run was made for this file. Every number is quoted
  from the sources below.

| short form | file | commit |
|---|---|---|
| `R §0`--`§7` | `agent_guidance/paper-v/overnight/closure-instability-20261010.md`, sections 0 to 7 (the linearized flow) | `40771435` |
| `R §8`--`§12` | the same report, sections 8 to 12 (escape statistics, predictor, confinement) | `c1c1f9cc` |
| `R §13` | the same report, section 13 (corrections after three refuters and a critic, and the ablations) | `36b686b3` |
| `R §14` | the same report, section 14 (the guard plus sparse repair hybrid) | `8390a83b` |
| `W` | `paper_5/notes/electron_phonon_closure_worklog_20260803.md`, blocks dated 2026-10-10 and 2026-10-11 | as above |

Where `R §8`--`§12` and `R §13` disagree, `R §13` holds, and section 7 below
lists the replaced sentence. Receipts are under
`output/local_runs/paper_v_instability_20261010/`, with `task2_corrections/`
for `R §13` and `task3_hybrid/` for `R §14`. The SHA-256 of every receipt
is in `sha256_task2.json` there (520 files after `R §14`; the manifest
file's own SHA-256 is
`767f6eec4c5facf11dc9948d88c3a9ad13907db8787bbbe98304ab88918d795f`).

**Configuration.** Unless a row says otherwise:

- the model is the Holstein dimer at `lambda_ep = 1.5`, `gamma = 0.5`
  (`omega_ph = 0.5`, `g = 0.6124`), drive amplitude 1, phonon cutoff 16;
- the "bare closure" is the unrepaired archive closure (`closed_scalar_rhs`,
  Eqs. (14a) to (14e) in the 31 closed scalar coordinates). Its blocks are
  the electronic one-body density matrix `rho` (3 coordinates), the
  coherent phonon amplitude `B` (4), the phonon covariance `N` (4), the
  anomalous phonon covariance `A` (6) and the electron--phonon correlation
  `C[q]` (14, of which 2 are the shared trace);
- the "`1e4` event" is the first time the largest absolute closed coordinate
  reaches `1e4`.

**The two preparations, never mixed.**

- The **Hartree--Fock preparation** is the Hartree--Fock electronic state
  with zero electron--phonon correlations: the record's divergent product
  state.
- The **exact correlated preparation** is the exact ground state of the
  truncated Hamiltonian.

Each has its own exact driven trajectory, and each is scored only against
its own:

- "the exact trajectory from Hartree--Fock" is the reference stored beside
  the repaired chain;
- "the exact trajectory from the exact correlated preparation" is the
  task-11 and task-13 reference.

**The accuracy metric.** "Accuracy over `[0, T]`" means the RMS over every
`0.0025`-grid sample in `[0, T]` against the exact trajectory from the same
preparation (`_accuracy_metrics`, `apcm_carried_witness_analysis.py`). It
is cumulative over the window, not the error at `T`. It is written as
`site-0 occupation / Eq. (22) energy / all 31 coordinates`.

**Escape tests.** An escape test hands a state to the bare closure. It uses
admissible perturbations: `x + eps u`, with `u` from
`constrained_initial_direction` (Frobenius norm 1). Such a `u` is tangent at
first order to the Eq. (22) energy and to Re and Im of the shared trace. No
positivity is imposed. The integrator is DOP853 at `rtol 1e-10`,
`atol 1e-12`.

---

## 1. The divergence

1. **Event times of the bare closure.**

   | start | `1e4` event | source |
   |---|---|---|
   | Hartree--Fock preparation, `t = 0` | `t = 54.4613`; 64 perturbations of size `1e-4` change it by at most `0.22` | `R §1`, `§3.3`, `§6` |
   | exact correlated preparation, `t = 0` | `t = 141.4057` | `R §0` item 1, `R §1` |
   | the exact trajectory from Hartree--Fock, its `t = 60` state | `t = 433.8964` at DOP853 `rtol 1e-10` (the record's value); `434.1811` at `1e-12`; `434.1878` at `1e-13`; RK4 at `0.00125` gives `434.1160` | `R §0` items 1 and 2 |
   | the repaired chain's `t = 60` state | `t = 410.7827` | `R §1` |

   - The exact correlated value corrects the record. The record's "never,
     through `t = 140`" (`divergent-initialization-20260920.md`
     section 2.2) was a window shorter than the event time.
   - The `t = 60` event is converged to about `0.07`. That is the size of
     the shift an admissible perturbation of order `1e-11` produces.
2. **The failure is not numerical.** Every event time converges under
   halving and quartering the step, under tightening the tolerance, and
   across an explicit and an implicit method, at all three starts
   (`R §0` item 2, `§3.1`).
3. **The flow preserves phase-space volume and the energy.**
   - The trace of the Jacobian vanishes at every state: at most `6.1e-13`,
     against a Jacobian 2-norm of up to `16.4` in the bounded phase and
     `2.8e4` at the event.
   - The 31 finite-time Lyapunov exponents sum to `-8.9e-8` over 374 units.
   - The Eq. (22) energy is conserved to six decimals through the `1e4`
     event (`R §0` item 3, `§6`).
   - So there is no attractor: the flow cannot contract volume.
4. **The divergence is transient chaos.** Along the bare closure from the
   exact trajectory's `t = 60` state (Hartree--Fock), the largest
   finite-time Lyapunov exponent over `t = 60` to 400 is `0.0468`
   (Benettin, step `0.0025`; `0.0463` and `0.0467` at steps `0.01` and
   `0.005`; `R §3.2`). The tangent-linear QR spectrum gives `0.047` at
   `T = 140`.
   - 64 perturbations of size `1e-4` fire after `293.1` to `478.1` units,
     median `351.7`, with none before `293` (`R §3.3`).
   - The bounded phase rides on a slow growth of the orbit's amplitude:
     the ten-unit maximum of the largest coordinate is `1.56`, `2.36` and
     `3.14` at `t = 60`, 250 and 400.
5. **The escape goes through coordinates the Eq. (22) energy does not
   contain.**
   - The energy is linear in the phonon covariance and in the diagonal of
     the electron--phonon correlation (`omega_ph Tr N` and
     `4 g Re sum_q C[q]_{qq}`). It does not contain the off-diagonal
     correlation entries.
   - Those entries are the ones that reach `1e4` on the four bare starts
     (`correlation_0_01_real`, `correlation_1_10_real`,
     `correlation_0_10_imag`, `correlation_0_01_imag`).
   - So the energy shell is not compact, and nothing in the closed
     equations confines those entries (`R §6`).
   - Past the event the largest coordinate goes from `1e4` to `1e10` in
     `0.25` to `0.33` time units on three of the four starts, the same at
     `rtol 1e-10` and `1e-12`. A finite-time singularity is not
     established (`R §6`).
6. **The growing direction is not the relative-mode phonon population.**
   The forward Lyapunov vector along the `t = 60` trajectory puts these
   shares of its squared norm on each block (`R §0` item 4):

   | block | share |
   |---|---:|
   | relative-mode phonon population | `0.045` |
   | anomalous phonon covariance | `0.355` |
   | electron--phonon correlation | `0.333` |

## 2. The cheap repair

The repaired chain has a fixed configuration:

- 27-key fixed dictionary, Hartree--Fock preparation with the exact-moment
  lift;
- Dykstra at 120 iterations onto both cones, every SSPRK(3,3) stage, at
  `h = 0.0025`;
- the Eq. (22) hyperplane at the drive-power reference;
- lift re-selection every 16 steps.

Task 11 is the same configuration from the exact correlated preparation,
with the declared-completion lift and the retry ladder, run to `t = 20`.

1. **Accuracy over `[0, T]`** (`R §13.6`, `cheap_repair_accuracy.json`):

   | route | T = 4 | T = 8 | T = 20 | T = 60 |
   |---|---|---|---|---|
   | repaired chain, Hartree--Fock | 0.0070 / 0.0011 / 0.0108 | 0.0331 / 0.0011 / 0.0372 | 0.0736 / 0.0012 / 0.1537 | 0.1604 / 0.0012 / 0.2523 |
   | bare closure, Hartree--Fock | 0.0522 / 0.0421 / 0.1296 | 0.0823 / 0.0440 / 0.2937 | 0.1682 / 0.0452 / 0.3320 | event at `54.46` |
   | task 11, exact correlated | 0.0116 / 0.0040 / 0.0093 | 0.0281 / 0.0049 / 0.0330 | 0.0681 / 0.0053 / 0.0560 | not run |
   | bare closure, exact correlated | 0.0296 / 0.0276 / 0.1109 | 0.0448 / 0.0222 / 0.1382 | 0.1431 / 0.0182 / 0.1932 | 0.3054 / 0.0162 / 0.3248 |

2. **Cost, one Hartree--Fock trajectory.** Three sessions measured
   different walls:

   | session | measurement | s per unit |
   |---|---|---:|
   | the recorded chain's own run (`R §13.2` (e)) | `12,990.087` s over `t = 0` to 60 | `216.5` |
   | a different run, reported by the refuter in `R §13.2` (e) | `t = 0` to 20 | `214.8` |
   | `R §13.11` | `t = 0` to `0.4`, minimum of three runs (`138.7`, `138.8`, `139.2` s) | `346.8` |
   | `R §14.4` | `t = 0` to `0.4`, one run of `132.8` s | `332.1` |

   In the `R §13.11` session, `117.2` of the `138.7` s went to 10 lift
   re-selections (8 recorded as failures), `18.8` s to 480 Dykstra repairs
   and `2.5` s to velocity evaluations.
3. **It does not remove the instability.** The test hands the state to the
   bare closure: 64 members, perturbation size `1e-4`, span 1500. All 64
   fire from every repaired state (`R §13.1`). Means in units after the
   state, with bootstrap 95 percent intervals (20,000 resamples):

   | state at | repaired chain | exact trajectory from Hartree--Fock | ratio, repaired to exact (unperturbed ratio) |
   |---|---|---|---|
   | `t = 20` | `588.7 [558.4, 622.0]` | `128.2 [127.1, 129.5]` | `4.59 [4.35, 4.85]` (`4.1`) |
   | `t = 40` | `482.6 [461.4, 506.2]` | `291.7 [275.5, 308.7]` | `1.65 [1.54, 1.78]` (`1.3`) |
   | `t = 60` | `410.0 [389.2, 433.9]` | `361.7 [351.0, 373.1]` | `1.13 [1.07, 1.21]` (`0.94`) |

   - The repaired mean falls with the state's time, by `106 [67, 146]`
     from `t = 20` to 40 and `73 [41, 104]` from 40 to 60.
   - At `t = 60` the unperturbed repaired state is not longer-lived than
     the exact one: `350.78` against `373.90`.
4. **The repair does not change the bare closure's linearization. It
   changes where that linearization is taken** (`R §0` item 5).
   - Over `t = 0` to 20 the leading real part of the Jacobian averages
     `0.177` along the repaired chain, `0.211` along the exact trajectory
     and `0.206` along the bare Hartree--Fock trajectory.
   - The difference is the growing direction. Along the bare trajectory it
     migrates into the electron--phonon correlation (share `0.31` to `0.57`
     from `t = 12` to 30). Along the repaired chain it stays in the phonon
     blocks (correlation share `0.05` to `0.23` from `t = 8`).

## 3. Representability as a predictor

"Representability" here is the retained 8-by-8 joint Gram of
`(1, b0, b1, b0^dag, b1^dag, sigma_x, sigma_y, sigma_z)`, with the shared
trace removed, being positive semidefinite (`run_unrepaired.py`). It is
measured by its minimum eigenvalue, "the Gram" (`R §13.3`).

1. **It is lost within units, from every state.**
   - Under the bare closure the Gram is negative at the start state, or
     within `0.5` to `12` units of it, in all 19 ensembles.
   - The exact trajectory's states start at `+3.8e-3`, `+7.5e-3` and
     `+9.9e-3`, and cross 0 within `1.5`, `0.75` and `1.2` units.
   - The Hartree--Fock preparation has a Gram of rank 4 of 8 at `t = 0`,
     and is below `-1e-8` by `t = 0.015`.
2. **It discriminates nothing.** In the task-11 `t = 20` ensemble, 38 of
   the 41 members with no event in 1500 units cross Gram `< -1`, at a
   median `542.8` units. The 23 members that fire cross at the same median
   `542.8`.
3. **Within one ensemble it does not order the events.** The first time
   below `-1` has rank correlation `-0.01` to `0.25` with the event time
   (64 members per ensemble).
   - Across the 19 starting states the medians correlate `0.905`
     (`p = 1.0e-7`). That is partly by construction, because every crossing
     precedes its event.
4. **A tuned lead.** The earliest threshold crossing that orders the
   members' events (rank correlation at least `0.9` over 64 members) comes
   a median number of units before the event:

   | quantity | thresholds | lead |
   |---|---|---|
   | the Gram | `-2.75` to `-4.5` | `18` to `38` units |
   | the largest coordinate | `2.4` to `4` | `12` to `26` units |

   - At equal predictive skill (`0.8`) the Gram is `1.3` to `6.2` units
     earlier.
   - Neither leads by more than about 40 units, which is under 10 percent
     of the `281` to `577` units to the event.
   - The scan covers 5 of 19 ensembles and 9 of 23 thresholds, and its
     thresholds were tuned on the same ensembles.

## 4. The guard: the second-order clip with the energy held

**What it holds** after every RK4 step (`R §13.4`):

- the Bloch ball, so the electronic `2 x 2` density is positive
  semidefinite;
- the phonon covariance positive semidefinite (`N_qq >= 0` and
  `|N_01|^2 <= N_00 N_11`);
- `|A_qq|^2 <= N_qq (N_qq + 1)` and the `|A_01|` bound;
- six correlation Cauchy--Schwarz bounds along the Pauli `x`, `y` and `z`
  axes;
- `Tr C[q] = 0`.

**What it does not hold:**

- the retained joint Gram. Over `t <= 60` it is below `-1e-4` at 64 to 75
  percent of unit records from Hartree--Fock (minimum `-0.366`), and 90 to
  98 percent from exact correlated (minimum `-0.63`);
- the boson moment matrix `[[N^T, A^*], [A, 1 + N]]` (minimum `-0.142` and
  `-0.302`);
- the Eq. (22) energy, unless the energy hold is added.

The map acts at every step: the fraction of member-steps clipped is `1.0`.

**The energy hold** (`R §13.6`). The map alternates the clip with the exact
root of `E(x + s grad E) = E_target`. `E_target` is the energy after the RK4
step, so the drive still does work on the state and the map does none. The
residual is at most `1.0e-12` in every run.

1. **Compactness, and its two conditions** (`R §13.2` (c)). Hold the Eq.
   (22) energy at `E <= E_max`, together with `N >= 0`, `|r| <= 1`, the
   twelve 2-by-2 Gram minors and `Tr C[q] = 0`. The reachable set is then
   compact, with the tight bound

   `E >= -2t + omega |B|^2 - 4 g |B| + omega Tr N - 2 g (sqrt N_00 + sqrt N_11)`.

   - That bound gives
     `|B| <= [4g + sqrt(16 g^2 + 4 omega (E_max + 2t + 2 g^2/omega))] / (2 omega)`
     and
     `sqrt N_qq <= [2g + sqrt(4 g^2 + 4 omega (E_max + 2t + 5 g^2/omega))] / (2 omega)`.
     The other blocks are bounded through these.
   - At `E_max = -3.5` the bounds are `|B| <= 4.90`, `N_qq <= 13.5`,
     `|A| <= 13.99` and correlations `<= 3.67`.
   - The `1e4` event needs `E >= 4.87e3`.
   - **The shared-trace pin is necessary.** The retained Gram subtracts
     `Tr C[q]`. With `N_00 = N_11 = n`, `x17 = -0.408 n` and all else zero,
     the energy is 0 and the Gram minimum is `+1`, yet `max|x| = n`. So the
     Gram held with the energy alone is not compact.
   - **The energy must be held exactly.** The clip alone heats. Over 3000
     units, 0 of 16 members fire, but the energy keeps rising from
     `t0 + 10`:

     | start | median energy change at 3000 | largest | largest coordinate |
     |---|---:|---:|---:|
     | exact correlated preparation | `+66.4` | `+319` | `285` |
     | exact `t = 60` state | `+3.9` | `+95.4` | `124` |

     Some members run away between 1200 and 1800 units (`R §13.9`).
2. **Which families are necessary.** 8 members per cell, perturbation size
   `1e-4`, 600 units, from both `t = 0` preparations (`R §13.5`).
   - **With the energy held exactly,** dropping any single family gives 0
     fired, with the largest coordinate at most `3.80`. No single family is
     necessary.
   - **The energy with the phonon bounds alone is not enough.** Keeping only
     the trace pin, only `N_qq >= 0`, both, or `N_qq >= 0` with the
     anomalous bounds lets 8 of 8 fire from both preparations.
   - Keeping the trace pin, `N_qq >= 0` and the anomalous bounds together
     confines from Hartree--Fock (0 fired) but not from exact correlated
     (8 fired).
   - Keeping only the correlation bounds gives 0 fired, but the largest
     coordinate reaches `169` at 600 from Hartree--Fock and is still
     growing.
   - **With the energy not held,** the correlation bounds are necessary.
     Without them 7 of 8 exact-correlated members fire (`502` to `576`
     units), and Hartree--Fock members reach `2.3e3`.
   - Dropping the anomalous bounds raises the post-pulse heating to `+190`
     (exact correlated) and `+7.8` (Hartree--Fock).
   - From exact correlated, the Bloch-ball clip is the source of the
     heating: without it the energy moves `+0.040` from `t = 10` to 600,
     against `+5.57` with it.
3. **A triggered guard** (`R §13.8`). The exact-energy clip is applied only
   where the largest coordinate exceeds 3 or the Gram falls below `-2.75`.
   - It prevents every event in 600 units: 0 of 16 from the repaired chain's
     `t = 60` state, 0 of 16 from the exact trajectory's.
   - The same 32 starts all fire under the bare closure, at `307` to `566`
     units.
   - It acts at `8.7e-4` and `9.7e-3` of member-steps.
   - **Its thresholds were tuned on the same ensembles** (`R §13.3`). No
     held-out test exists.
4. **Cost.** One trajectory per preparation, one session each:

   | session | bare RK4 step | clip | clip with the energy hold |
   |---|---|---|---|
   | `R §13.11`, Hartree--Fock | `0.140` s per unit | `0.165` (`1.18` times bare) | `0.216` (`1.55` times) |
   | `R §13.11`, exact correlated | `0.139` | `0.164` (`1.18` times) | `0.426` (`3.07` times) |
   | `R §14.4`, Hartree--Fock | `0.130` | not measured | `0.203` |
   | `R §14.4`, exact correlated | `0.132` | not measured | `0.408` |

   The clip is about 21 square roots and 22 comparisons per state per
   step.
5. **First order in `h`** (`R §13.4`). With a map after every RK4 step, the
   ratio of successive step-halving differences is `1.91` to `2.39` at
   `t = 20`, against `16.0` for the bare closure.
   - The scores below move by at most `2.4e-3` between `h` and `h/2` at
     `T <= 20`, and by at most `2e-3` at `T = 60`.
   - The exceptions are the Gram cone with exact energy (not
     `h`-converged; scores move up to `0.041`) and the coherent-field
     restoration from exact correlated at `T = 60` (`0.031`).
6. **Accuracy: the guard buys nothing on the observables** (`R §13.6`; one
   unperturbed trajectory, scores over `[0, T]`).

   | map, Hartree--Fock | T = 4 | T = 20 | T = 60 |
   |---|---|---|---|
   | bare closure | 0.0522 / 0.0421 / 0.1296 | 0.1682 / 0.0452 / 0.3320 | event at `54.46` |
   | clip | 0.0356 / 0.0436 / 0.1313 | 0.1654 / 0.0480 / 0.3145 | 0.2715 / 0.0858 / 0.3576 |
   | clip with the energy hold | 0.0356 / 0.0437 / 0.1313 | 0.1655 / 0.0487 / 0.3146 | 0.2890 / 0.0495 / 0.3668 |
   | Gram cone (one eigenvalue clip per step) | 0.0359 / 0.2292 / 0.1158 | 0.0859 / 0.3841 / 0.2584 | 0.1979 / 0.4902 / 0.3048 |
   | boson moment matrix plus clip, energy held | 0.0448 / 0.0409 / 0.1057 | 0.1611 / 0.0439 / 0.2651 | 0.2609 / 0.0443 / 0.3042 |
   | triggered guard | 0.0522 / 0.0421 / 0.1296 | 0.1682 / 0.0452 / 0.3320 | 0.3155 / 0.0457 / 0.4294 |
   | cheap repair (repaired chain) | 0.0070 / 0.0011 / 0.0108 | 0.0736 / 0.0012 / 0.1537 | 0.1604 / 0.0012 / 0.2523 |

   | map, exact correlated | T = 4 | T = 20 | T = 60 |
   |---|---|---|---|
   | bare closure | 0.0296 / 0.0276 / 0.1109 | 0.1431 / 0.0182 / 0.1932 | 0.3054 / 0.0162 / 0.3248 |
   | clip | 0.0269 / 0.1008 / 0.1081 | 0.1586 / 0.1603 / 0.1686 | 0.2810 / 0.3070 / 0.2559 |
   | clip with the energy hold | 0.0192 / 0.0352 / 0.1059 | 0.1347 / 0.0353 / 0.1537 | 0.3004 / 0.0354 / 0.2513 |
   | cheap repair (task 11) | 0.0116 / 0.0040 / 0.0093 | 0.0681 / 0.0053 / 0.0560 | not run |

   - **On the energy,** no map comes within `6.6` times the cheap repair's
     Eq. (22) RMS at any horizon. The best map is `29` to `38` times worse
     from Hartree--Fock and `6.6` to `8.7` times from exact correlated.
   - **On the site-0 occupation,** the cheap repair is better at `T = 4`
     and 8 from both preparations, and at `T = 20` from exact correlated.
     The one exception is the Gram cone with exact energy from Hartree--Fock
     at `T = 60` (`0.1285` against `0.1604`), which is not `h`-converged.
   - **The remaining energy error.** What is left of the Eq. (22) RMS under
     the hold (`0.035` exact correlated, `0.044` to `0.050` Hartree--Fock)
     is the difference between the work the drive does on the closure and
     on the exact state. At `t = 1` the closure's energy is `-3.567` with
     or without a map, against the exact `-3.597`.
   - **The critic's assessment, verbatim** (`R §13.4`): *the minors clip is
     a candidate cheap divergence guard, not a representability repair, and
     it does not improve accuracy.*
7. **Representable starts escape as before** (`R §13.10`). The repaired
   chain's and the exact trajectory's `t = 20` states were perturbed (16
   members, sizes `1e-4` and `1e-3`), carried into the clip's set or the
   Gram cone, then handed to the bare closure. All 128 fire:

   | start | mean time to the event | section-9 mean, unprojected (64 members) |
   |---|---|---:|
   | repaired chain | `420` to `541` | `588.7` |
   | exact trajectory | `125.8` and `130.3` | `128.2` |

   Escape is a property of the flow, not of where the start sits relative
   to the cones.

## 5. The hybrid: the guard every step, the repair every `Delta` units

**Definition** (`R §14.1`). The bare closure runs with the exact-energy clip
after every RK4 step at `h = 0.0025`. At `t = Delta, 2 Delta, ..., 60` one
recorded repair step is applied:

1. the 29 raw moments are taken from the closure state;
2. the 27 dictionary moments the repair needs, and the closure does not
   carry, are set by the zero-cumulant rule;
3. the lift is re-selected by the recorded rule;
4. one Dykstra-120 repair is applied onto both cones, with the Eq. (22)
   hyperplane at the energy carried in;
5. the result is mapped back, and the clip applied once more.

**The zero-cumulant rule is an assumption of this lane.** It equals the
exact-moment dictionary of the Hartree--Fock product state to `4.4e-16`. It
differs from the exact correlated ground state's dictionary by up to
`0.378`.

The tables use accuracy over `[0, T]`. "s per unit" is the arm's own wall
in one serial session (`R §14.4`). The escape column comes from the arm's
`t = 60` state under the bare closure: 8 members, size `1e-4`, span 600. It
gives the number fired by 600, the restricted mean to 600 (a lower bound
when members are censored) and the unperturbed state's own event.

**Hartree--Fock** (bare step `0.130` s per unit, clip with the energy hold
`0.203`):

| `Delta` | T = 4 | T = 20 | T = 60 | s per unit | Gram minimum; fraction below `-1e-4` | escape |
|---|---|---|---|---:|---|---|
| 1 | 0.0318 / 0.0277 / 0.1364 | 0.1420 / 0.0288 / 0.2695 | 0.2506 / 0.0289 / 0.2919 | 3.258 | -0.200; 0.074 | 0; 600.0; none |
| 2 | 0.0370 / 0.0432 / 0.1392 | 0.1528 / 0.0477 / 0.2906 | 0.2600 / 0.0484 / 0.3063 | 1.711 | -0.354; 0.216 | 0; 600.0; none |
| 5 | 0.0356 / 0.0437 / 0.1313 | 0.1603 / 0.0487 / 0.2966 | 0.2666 / 0.0495 / 0.3211 | 0.778 | -0.381; 0.400 | 0; 600.0; none |
| 10 | 0.0356 / 0.0437 / 0.1313 | 0.1638 / 0.0487 / 0.3020 | 0.2782 / 0.0495 / 0.3335 | 0.480 | -0.381; 0.809 | 8; 481.4; none |
| 20 | 0.0356 / 0.0437 / 0.1313 | 0.1655 / 0.0487 / 0.3146 | 0.2852 / 0.0495 / 0.3570 | 0.357 | -0.381; 0.863 | 8; 227.9; 223.0 |
| infinity (guard alone) | 0.0356 / 0.0437 / 0.1313 | 0.1655 / 0.0487 / 0.3146 | 0.2890 / 0.0495 / 0.3668 | 0.224 | -0.381; 0.750 | 8; 368.8; 405.4 |
| every stage (repaired chain) | 0.0070 / 0.0011 / 0.0108 | 0.0736 / 0.0012 / 0.1537 | 0.1604 / 0.0012 / 0.2523 | 332.1 (`R §14.4` session); 216.5 (recorded) | -0.00085; 0.242 | 8; 373.6; 350.8 |

**Exact correlated** (bare step `0.132`, clip with the energy hold `0.408`):

| `Delta` | T = 4 | T = 20 | T = 60 | s per unit | Gram minimum; fraction below `-1e-4` | escape |
|---|---|---|---|---:|---|---|
| 1 | 0.0225 / 0.0532 / 0.1289 | 0.0927 / 0.0599 / 0.1579 | 0.3788 / 0.0610 / 0.3281 | 3.081 | -0.197; 0.609 | 3; 584.6; none |
| 2 | 0.0280 / 0.0377 / 0.0900 | 0.0918 / 0.0407 / 0.1612 | 0.3710 / 0.0412 / 0.3240 | 1.623 | -0.195; 0.889 | 4; 512.2; 585.9 |
| 5 | 0.0192 / 0.0352 / 0.1059 | 0.0575 / 0.0353 / 0.1047 | 0.2857 / 0.0354 / 0.2441 | 0.804 | -0.433; 0.414 | 0; 600.0; none |
| 10 | 0.0192 / 0.0352 / 0.1059 | 0.1200 / 0.0353 / 0.1463 | 0.2131 / 0.0354 / 0.1975 | 0.529 | -0.749; 0.997 | 0; 600.0; none |
| 20 | 0.0192 / 0.0352 / 0.1059 | 0.1347 / 0.0353 / 0.1537 | 0.2725 / 0.0354 / 0.2459 | 0.714 | -0.749; 0.986 | 0; 600.0; none |
| infinity (guard alone) | 0.0192 / 0.0352 / 0.1059 | 0.1347 / 0.0353 / 0.1537 | 0.3004 / 0.0354 / 0.2513 | 0.353 | -0.749; 0.974 | 6; 504.6; 478.5 |
| every stage (task 11, to `t = 20`) | 0.0116 / 0.0040 / 0.0093 | 0.0681 / 0.0053 / 0.0560 | not run | not measured | --- | no `t = 60` state |

1. **The energy gap.** The best arm (Hartree--Fock, `Delta = 1`) is `24`
   times worse than the repaired chain on the Eq. (22) energy over
   `[0, 60]` (`0.0289` against `0.0012`). It is `1.56` times worse on the
   site-0 occupation and `1.16` times on all 31. Over `[0, 4]` it is `4.5`
   and `26` times worse on the occupation and the energy.
2. **From exact correlated the sparse repair does not improve on the
   guard.** At `T = 60`, `Delta = 1` and 2 are worse than the guard alone,
   and repairs during the pulse raise the Eq. (22) RMS to `0.0610`. The one
   cell below task 11 (`Delta = 5`, site-0 over `[0, 20]`: `0.0575` against
   `0.0681`) has neighbours at `0.0918` and `0.1200`. It is not a trend.
3. **Over the guard, from Hartree--Fock,** the scores improve steadily as
   `Delta` shrinks. `Delta = 1` lowers the `T = 60` scores by 13, 42 and 20
   percent, at `14.5` times the guard's wall.
4. **The lift re-selection costs most of the wall and does nothing.** It
   returned "infeasible" in all 222 events, so the carried lift was always
   used. It takes `2.4` to `3.0` s of each event's `2.5` to `3.1` s; the
   repair call itself takes `0.04` s.
   - The same trajectories would therefore cost `0.22` to `0.29` s per
     unit with the re-selection skipped. That figure is derived from the
     same run, not measured.
   - With the re-selection, the hybrid is 100 to 930 times cheaper than the
     cheap repair's `332.1` s per unit in the same session.
5. **The escape test does not order with `Delta`.** From Hartree--Fock, 0 of
   8 fire by 600 at `Delta = 1`, 2 and 5, while all 8 fire at `Delta = 10`
   and 20 and for the guard alone. From exact correlated the counts are 3,
   4, 0, 0, 0 and 6 of 8 for `Delta = 1`, 2, 5, 10, 20 and infinity.
6. **The repair does not reach its cone.** The Dykstra repair at 120
   iterations never reaches the extension cone: the minimum after every
   event is `-1.9e-4` to `-4.1e-3`.
7. **Step dependence.** The `h/2` check on Hartree--Fock `Delta = 5` moves
   the scores by at most `1.8e-4`. The other arms were not checked.

## 6. Reading for the paper

The manuscript asks (its placeholder for the high-`U` failure mode) whether
the failure is density-matrix positivity loss, stiffness, correlation
blow-up, energy drift or self-consistency instability. In its own
vocabulary: the equal-time non-Markovian equations of motion for `rho`, the
phonon covariance, the anomalous phonon covariance, `B` and `delta rho_12^q`.

- **What the archive closure does.** It is fast: one trajectory costs
  `0.13` to `0.14` s per time unit on this host. Its failure is not energy
  drift and not stiffness.
  - The total energy after the external pulse is conserved to six decimals
    through the blow-up.
  - The trace of the flow's Jacobian vanishes.
  - The one-step backward-Euler matrix at `dt = 0.0025` has condition
    number `1.008` to `1.038` wherever the largest coordinate is below 5
    (`R §0` item 6).
- **What breaks.**
  - The failure mode is correlation blow-up: unbounded growth of the
    off-diagonal entries of `delta rho_12^q`, which the energy does not
    contain. It is preceded by transient chaos (largest exponent about
    `0.047`).
  - It happens from every state tested: the Hartree--Fock preparation at
    `t = 54.46`, the exact correlated preparation at `141.41`, and exact
    and repaired states handed over at `t = 20` to 60, within hundreds of
    units.
  - Positivity loss comes first and within units. The joint electron--boson
    second moments stop being representable within `0.5` to 12 units of
    every start measured, long before any blow-up. That loss does not
    predict which states blow up.
- **What can be held cheaply.** The clip keeps:
  - `rho` positive semidefinite;
  - the phonon covariance positive semidefinite;
  - the anomalous-covariance and electron--phonon Cauchy--Schwarz bounds;
  - the trace of `delta rho_12^q` at zero;
  - with the energy hold, the total energy exactly.

  Together these confine the state to a compact set, at `1.2` to `3.1`
  times the cost of a step. That is confinement without accuracy. Over
  `[0, 20]` the clip with the energy hold changes the site-0 occupation and
  total-energy RMS by at most `0.017`, in either direction, from the
  unrepaired closure's.
- **What accuracy costs.** The Dykstra repair (section 2 of this file) gets
  the site-0 occupation within `0.0070` and the total energy within
  `0.0011` RMS over the first 4 units from Hartree--Fock. It costs three
  orders of magnitude more per unit: `216.5` to `346.8` s, against `0.13`
  to `0.22` for the closure and the clip. It does not remove the
  instability: every repaired state blows up once handed back to the
  closure.
- **No measured combination bridges the two.** Repairing every 1 to 20
  units on top of the clip stays 24 times worse on the total energy, and
  does not improve on the clip from the exact correlated preparation.

## 7. Claims withdrawn during verification, with their replacements

Each line gives the withdrawn wording, then the replacement and where it is
recorded.

1. **"Never, through `t = 140`"** (exact correlated preparation, bare
   closure; `divergent-initialization-20260920.md` §2.2) → fires at
   `t = 141.4057` (`R §0`).
2. **"Repaired states escape at a rate about five times lower than the bare
   closure's own states"** (`R §8`) → it compared two preparations. Against
   the exact trajectory from the same preparation the repaired states live
   `4.6`, `1.65` and `1.13` times longer at `t = 20`, 40 and 60
   (unperturbed `4.1`, `1.3`, `0.94`) (`R §13.0`, `§13.1` (c)).
3. **"The bare closure's own states take `136.0`, `104.8`, `82.7`"** → those
   are one deterministic trajectory's remaining lifetime `141.41 - t`, plus
   `+14.6`, `+3.4`, `+1.2` from the perturbation (`R §13.1` (a)).
4. **"Censored maximum-likelihood rates"** → inverse mean times,
   `k / exposure`. Censoring enters 1 of 19 rows (`R §13.1`).
5. **"Admissible" read as representable** → three linear equalities only.
   The base states already violate the electronic, phonon and joint cones
   (`R §13.1` (b)).
6. **"Five of nineteen ensembles fixed to within a unit"** → four of
   nineteen have all 64 events within one unit (`R §13.1` (e)).
7. **The after-delay rate left blank under a spread rule** → given for every
   complete row (`R §13.1` (f)).
8. **Task 11 at `t = 20`, rate `2.55e-4`** → restricted mean to 1500 is
   `1409.7` (a lower bound), the inverse mean at most `7.09e-4`, and the
   Kaplan--Meier median above 1500 (`R §13.1` (g)).
9. **The section-9 contrast "repaired chain against states the repair held
   cones on"** → task 11 and the repaired chain carry the same repair
   configuration. They differ in preparation and extension backend
   (`R §13.1` (h)).
10. **"Task 13 hands on the shortest-lived states" and "task 11 the
    longest-lived of all"** → these hold only among the three repair
    chains. Task-11 times are erratic (`413.5`, `199.2`, `1038.1`, above
    1500) (`R §13.2` (a)).
11. **"The off-diagonal Cauchy--Schwarz bound lowers the escape rate `1.0`
    to `7.6` times; the escape moves to the phonon blocks"** → it delays
    escape `5.2` times (Hartree--Fock), at least `3.6` times (exact
    correlated, restricted mean), `1.25` to `1.29` times (exact `t = 60`,
    log-rank `p = 4e-9`) and not detectably (repaired `t = 60`). The escape
    is not confined to the phonon blocks (`R §13.2` (b), `§13.4`).
12. **The compactness bound with `4 sqrt 2 g`; "the retained joint Gram is
    sufficient with the energy"; "each part is measured"** → the tight
    bound of section 4 above. The Gram with the energy alone is not
    compact, because the trace pin is necessary. Necessity was measured
    only in `R §13.5` (`R §13.2` (c)).
13. **"No member escapes, with or without the energy restored" read as
    confinement evidence** → no run before `R §13` held the energy exactly.
    The restoration through `B` failed (largest change `40.99`). Zero
    escapes followed from the measured heating (`R §13.2` (c)).
14. **"Every map above confines"** → only the clip and the clip with the
    coherent-field restoration. The off-diagonal bound alone does not
    (`R §13.2` (d)).
15. **"The clip costs about twice the bare step (`0.20` against `0.46` s)"**
    → `1.18` times, in one session. The two walls were separate runs under
    different load (`R §13.2` (e), `§13.11`).
16. **Accuracy "at `t = 20`"** → cumulative RMS over `0 <= t <= 20`
    (`R §13.2` (f)).
17. **"128 member-runs"** → 64 distinct initial conditions, each run under
    two maps (`R §13.2` (h)).
18. **"Every second-order bound" and "positivity bounds on every block"** →
    the explicit list in section 4. The clip holds neither the joint Gram,
    nor the boson moment matrix, nor the energy (`R §13.4`).
19. **Pulse inputs `0.22` (Hartree--Fock) and `0.24` (exact correlated)** →
    swapped: `0.24` and `0.22` (`R §13.4`).
20. **The off-diagonal bound from exact `t = 60`, exponential rate ratio
    `1.29`** → median `366` to `469`, log-rank `p = 4e-9` (`R §13.4`).
21. **"Task 13 faster as it heats"** → its states escape at absolute
    `t = 83.8` to `86.2`, nearly fixed (`R §13.4`).
22. **"Loses joint representability long before its event"** → negative at
    the start or within `0.5` to 12 units, in all 19 ensembles. "Below `-1`"
    is a depth threshold, not loss (`R §13.3`).
23. **"Carries no information about when the event comes"** → within an
    ensemble the rank correlation is `-0.01` to `0.25`. Across the 19
    starting states it is `0.905`, partly by construction (`R §13.3`).
24. **"No earlier and no better than the coordinate magnitude passing 4"** →
    the tuned Gram leads by 18 to 38 units, against 12 to 26 for the
    magnitude (`R §13.3`).
25. **The early crossings are "unrelated"** → they order the events weakly
    (rank correlation `-0.06` to `0.42`; `0.42` and `0.31` distinguishable
    from zero) (`R §13.3`).
26. **"The loss predicts that the bare closure will escape"** → withdrawn.
    No contrast case exists (`R §13.3`).
27. **"Bare `t = 20` value `-1.63` at 200 before"** → that rests on 4 of 64
    members. The 64-member value at 100 before is `-0.99` (`R §13.3`).
28. **The `1.1e-16` margin agreement** → it holds on bounded states. On the
    bare Hartree--Fock trajectory it is `6e-16` relative (`R §13.3`).

## 8. What remains unmeasured, in priority order

Each line names the bound the measurement would take, estimated from the
walls measured above. These are estimates, not runs. Every process stays
under 20 minutes.

1. **Confinement by the clip with the energy hold beyond 600 units.** This
   decides whether the compact set is reached in practice. It would take 8
   members to 3000 units in 600-unit batches, each batch 3 to 8 minutes
   (the 600-unit cells took `159` to `503` s).
2. **What makes the cheap repair accurate.** The candidates are the
   dictionary dynamics, the retained-Gram target, the 36-row positive
   extension and the hyperplane. Each candidate needs one every-stage chain
   with one ingredient removed, to `t = 20`: about `1.2` to `1.9` hours at
   `216.5` to `346.8` s per unit, in segments under 20 minutes.
3. **The exact correlated every-stage reference to `t = 60`** with the
   repaired chain's driver. This is about `3.6` to `5.8` hours in segments.
   It gives the hybrid table its missing reference.
4. **The hybrid with another dictionary rule,** for example carrying the
   repair's own dictionary between events. Each arm took at most `3.3`
   minutes.
5. **The hybrid at `Delta` below 1, with the re-selection skipped and run
   directly.** This would cost about `0.29` s per unit at `Delta = 0.5`
   and `0.37` at `0.25` (derived), so minutes per arm.
6. **A converged Dykstra repair inside the hybrid.** The iteration count is
   unknown. At 120 iterations a repair takes `0.04` s.
7. **A held-out test of the triggered guard's thresholds,** on starts not
   used to choose them. This needs 16 members per start at about 3 to 6
   minutes per 600 units.
8. **Escape statistics with more members and a longer span** for the
   hybrid's and the guard's hand-on states. 64 members to 1500 units takes
   minutes with 10 processes (one DOP853 member takes 10 to 20 s per
   several hundred units).
9. **Accuracy over perturbed members** rather than one unperturbed
   trajectory per map. It costs seconds per member per map.
10. **The repaired chain with the repair switched off at `t = 20`, scored
    for accuracy.** The bare closure costs `0.13` s per unit, so seconds.
11. **Repaired states other than `t = 20`, 40 and 60 of one chain,** and
    other chains. Each 64-member ensemble takes minutes.
12. **The step check `h/2`** for the hybrid arms other than Hartree--Fock
    `Delta = 5`, and for the exact correlated preparation. About 1 minute
    per arm.
13. **The count of non-converged map steps** in the batched exact-correlated
    drop-trace cell, where only the energy residual was recorded. Seconds,
    with a counter added.
