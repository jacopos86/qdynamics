# Route (a) priced end to end: the re-centred windowed order-3 lift against the bare closure (2026-10-11)

Date: 2026-10-11. Author: the persistent Paper V quantum-integration agent
(Claude).

**What it answers.** The coordinator's task after revision 2 (committed at
`475c47a7`): price route (a) end to end and set route (b) beside it.

**Status.**

- Branch `claude/paper-v-routes-20260914`, HEAD `5886455e` at writing.
- Exploratory, not promoted.
- Nothing committed by this agent; no manuscript edit; no default changed.
- Receipts are under `output/local_runs/paper_v_quantum_target_form_20261010/r3/`,
  with their hashes in section 7.

**Revision note (2026-10-18, after root reviews 027 and 030).**

- **What this revises.** The version committed at `d3c80e8a` (307 lines,
  SHA-256 `e98c046f426d4a668a907e9f7002bad04e9c6ab07ddefd27262ab96a675ddee3`),
  which stays in history.
- **What prompted it.** Root review
  `quantum-r2-pricing-review-20261011-027.md` (SHA-256 `3e6ecfc0...`), and
  the quantum part of `consolidated-results-review-20261011-030.md`
  (SHA-256 `14c14fd4...`).
- **What changed.** 35 statements: 24 corrected and
  11 added. Each finding is dispositioned in
  `DISPOSITION-quantum-027-030-claude-20261010.md` beside this file.
- **Where the new numbers come from.** Arithmetic on the stored receipt
  `r3_ledger_sigma2e-4.json`, in
  `output/local_runs/paper_v_quantum_target_form_20261010/r4_flow_budget_review/r4_ledger_recount.json`.
  No trajectory, solver or noise model was run.
- **How to read the table.** Line numbers are in this revised file.

| line | section | statement now (was) | finding |
|---|---|---|---|
| 75 | 0 | Four marginal p95 at `sigma = 2e-4` in 21 batches; standard deviation, not a guarantee; proxies `5.6e8` to `6.8e8` (LCU) and `4.9e6`; neither validated (was: 20 readouts "accurate to `sigma <= 2e-4`", `1.1e9`) | 027-1, 027-2, 030-2 |
| 114 | 1 | Added: the output readouts are independent of the restart readout, so `R + 2` batches, 21 at `R = 19` | 027-2 |
| 125 | 1 | Added: the shots model is a Gaussian surrogate with the true norm (`r3_route_a.py:77-86`) | 027-2 |
| 142 | 1 | Added: four separate empirical marginal percentiles, not a simultaneous statement | 027-2 |
| 177 | 2 | Among the tested levels; finite comparison of a surrogate; not necessary or sufficient (was: "the largest admissible `sigma`") | 027-2, 030-2 |
| 186 | 2 | 21 readout batches; worst-case errors add linearly (was: 20 readouts) | 027-2 |
| 191 | 2 | Crossing by interpolation; `8.5e10` window solves over 21 batches (was: `8e10` over 20) | 027-2 |
| 193 | 2 | Amplitude estimation `pi ||Y_1|| / epsilon_coord`, `8.4e3` to `2.7e4` (was: `pi ||x|| / sigma`, `3.7e4`) | 027-1 |
| 209 | 3 | Readout batches 7, 12, 21, 42 (was: 6, 11, 20, 41) | 027-2, 030-2 |
| 222 | 3 | 42 batches at `R = 40` (was: 41) | 027-2 |
| 225 | 3 | A finite comparison of four counts, not a unique optimum or a monotone law | 027-2 |
| 251 | 4 | Added: sparse `alpha_BE` is measured at `+f_max` only; the family bound needs both endpoints | 027-3, 030-3 |
| 252 | 4 | Added: the LCU value is a candidate triangle bound with no circuit | 027-3 |
| 255 | 4 | `beta_j` frozen at `f_max`; the uniform numerator is 1.13 to 1.62 times larger; neither is bound to a theorem | 027-3, 030-3 |
| 258 | 4 | The published readout row is labelled "as first published" | 027-1 |
| 259 | 4 | Added: corrected readout `31 pi ||Y_1,j|| / epsilon_coord`; `||Y_1||/gamma` = 0.235 to 0.692; `1.20e7` over 21 batches | 027-1, 030-2 |
| 266 | 4 | Totals `5.6e8` / `7.7e9` corrected, `6.8e8` / `9.5e9` with uniform forcing, beside the published `1.1e9` / `1.5e10` | 027-1, 027-3 |
| 270 | 4 | Proxies, not end-to-end costs; omissions expanded: time-ordered propagator, driven-family access, forcing state, discretization, norm/sign/confidence (was: three omissions) | 027-1, 027-3, 030-3 |
| 282 | 4 | Added: the deterministic readout contract written out: tolerance split, `98.6 / theta`, norm, sign, and why `2e-4` is not shown sufficient | 027-2, 030-2 |
| 314 | 4 | Readout count `1.20e7` corrected (was: `2.4e7`) | 027-1, 027-2 |
| 321 | 4 | Added: one same-target benchmark, not a lower bound or a no-speedup theorem | 027-3, 030-3 |
| 324 | 4 | No speedup "on these proxies at this size"; full-coordinate readout is not automatically the classical cost (was: "plainly ... no speedup"; "the classical cost itself") | 027-3, 030-7 |
| 343 | 5 | Added: the whole `1e-2` goes to readout; counts scale as `1e-2 / epsilon_read` | 027-4 |
| 344 | 5 | Route (b) total labelled a leading-factor proxy | 027-4, 030-2 |
| 345 | 5 | Added: omitted `alpha_H`, ground-state preparation, simulation error, failure probability `4.38` | 027-4 |
| 346 | 5 | Cutoff 16 against 20 are computed differences of finite models (was: "cutoff error") | 027-4, 030-4 |
| 347 | 5 | Model is the finite cutoff-16 Hamiltonian, its distance from infinite cutoff not bounded (was: "none") | 027-4, 030-4 |
| 354 | 5 | Route (b) own-target error not computed for an implemented route; the cutoff difference is labelled as such; route (a) row marked Gaussian surrogate (was: "cutoff `2.6e-4`" as its error) | 027-4, 030-4 |
| 355 | 5 | Error against T-physical bounded by `max(0, Delta - epsilon)` and `Delta + epsilon`; route (b) not certified (was: "adds the closure's") | 030-4 |
| 357 | 5 | Side-by-side totals as proxies, corrected and published | 027-1 |
| 360 | 5 | Ratio about 230 as published, 110 to 140 corrected; not an established comparison; no route selected (was: "230 times fewer") | 027-1, 030-2 |
| 371 | 6 | `C_max` is for the frozen generator; the driven case needs the time-ordered propagator | 027-3 |
| 374 | 6 | Readout omissions: `||Y_1||`, and a union bound over 651 estimates, `9.47` / `9.51` (was: `||x||`, 620, `9.4`) | 027-2, 030-2 |
| 384 | 6 | Added: time-ordered propagator, driven-family access and forcing-state preparation unmeasured | 027-3 |
| 400 | 7 | Per-process timers only; the whole increment was not timed (was: "each process ran in under 20 minutes") | 027 timing |

## 0. The one sentence

In the executed Gaussian-surrogate protocol, route (a)'s four empirical
marginal 95th percentiles stay within `1e-2` of the bare closure when each
of its 21 readout batches of all 31 coordinates (19 restarts, plus
independent outputs at `T = 4` and `20`) carries noise of standard
deviation `sigma = 2e-4` per coordinate. A standard deviation is not an
absolute-error or simultaneous-confidence guarantee. Its leading-factor
proxy is about `5.6e8` to `6.8e8` solver calls with LCU access (`1.1e9` as
first published), set beside one `0.19 s` classical integration. Route (b)'s
proxy against the exact cutoff-16 dimer is about `4.9e6` calls on 12 qubits.
Both proxies omit costs listed in sections 4 to 6. Neither is a validated
gate, runtime or total-error cost.

---

## 1. Definitions used throughout

**Target.** T-bare is the bare archive closure, `closed_scalar_rhs` in the 31
closed coordinates, integrated from the exact correlated preparation (the
cutoff-16 ground state, `lambda_ep = 1.5`, `gamma = 0.5`, drive amplitude
1). The integrator is DOP853 at `rtol 1e-12`, `atol 1e-13`, sampled every
`0.0025` (`task1_trajectories.npz`). Every route (a) number below is scored
against T-bare.

**Route (a).**

- **Windows.** The time axis `t = 0..20` is cut into `R + 1` windows of equal
  step count. `R` is the number of re-lifts, the restarts after the initial
  lift. With `R = 19`, the windows have length 1.
- **At each restart.** The 31 coordinates `x_r` are read out. The order-3
  Carleman truncation of `y = x - x_r` about `x_r` is built, with forcing
  `F(t, x_r)`. It is integrated from `y = 0` across the window by classical
  RK4 at `h = 0.01`. The window's end state `x_r + Y_1` is the next readout.
- **The initial state** `x(0)` is exact.
- **Integrator check (`r3_check.json`).** The batched lift velocity matches the
  module's `carleman_block_derivative` to `1.2e-16` relative. The noise-free
  result at `h = 0.01` reproduces revision 2 calculation 1 (`h = 0.0025`) to
  all four printed digits.

**Noise models.** Noise is applied to every restart readout and to the
readout at the measurement times `T = 4` and `20`. The output readouts are
draws independent of the restart readout at `t = 4` (`r3_route_a.py:156`
and `:161`). So `R` re-lifts mean `R + 2` readout batches of 31
coordinates: 21 at `R = 19`.

- **gauss:** `x_read = x + sigma xi`, with `xi ~ N(0, I_31)`; `sigma` is per
  coordinate, in closed-coordinate units.
- **shots:** each normalized amplitude `a_i = x_i / ||x||` is estimated from
  `S` two-outcome shots (a Hadamard-test estimate, variance
  `(1 - a_i^2) / S`), with `||x||` assumed known. The per-coordinate standard
  deviation is therefore `||x|| sqrt((1 - a_i^2)/S)`, about
  `2.3 / sqrt(S)` at `||x|| = 2.28`. The driver draws Gaussian noise of that
  variance with the true `||x||` (`r3_route_a.py:77-86`). This is a Gaussian
  surrogate, not executed binomial sampling or amplitude estimation.

**Observables and errors.**

- The observables are the site-0 occupation per spin `(1 + x_0)/2` and the
  Eq. (22) energy `H(x)`.
- The error of each is `|O(route) - O(T-bare)|` at `T`.
  - The **propagated** error has a noise-free final readout: it is the
    effect of the restart noise alone.
  - The **contract** error has a noisy final readout, read with the same
    noise model.
- 50 noise draws per row, integrated as one batch. Each row gives the mean,
  median and 95th percentile over draws.

**The contract.** The 95th percentile of the contract error is at most
`1e-2` for both observables at both horizons. These are four separate
empirical marginal percentiles over 50 draws, not a simultaneous 95 percent
statement.

## 2. Readout noise through the re-centred lift

Order 3, `R = 19` re-lifts (windows of length 1), 50 draws per row. Receipts:
`r3_noise_gauss.json`, `r3_refine_gauss.json`, `r3_noise_shots.json`,
`r3_refine_shots.json`.

Cells give the contract error as mean / median / 95th percentile. The
noise-free values (order 3, `R = 19`, final readout exact) are, for site0 and
energy:

| horizon | site0 | energy |
|---|---|---|
| `T = 4` | `2.6e-5` | `8.4e-5` |
| `T = 20` | `1.6e-4` | `1.2e-4` |

| noise | site0 `T=4` | site0 `T=20` | energy `T=4` | energy `T=20` | contract holds |
|---|---|---|---|---|---|
| none | `2.6e-5` | `1.6e-4` | `8.4e-5` | `1.2e-4` | yes |
| gauss `1e-4` | `1.2e-4` / `1.1e-4` / `3.0e-4` | `1.2e-3` / `1.1e-3` / `3.1e-3` | `8.1e-4` / `6.6e-4` / `1.8e-3` | `2.1e-3` / `2.1e-3` / `4.6e-3` | yes |
| gauss `2e-4` | `2.5e-4` / `2.1e-4` / `5.8e-4` | `2.5e-3` / `2.1e-3` / `6.3e-3` | `1.6e-3` / `1.4e-3` / `3.6e-3` | `4.2e-3` / `4.0e-3` / **`9.2e-3`** | **yes** |
| gauss `3e-4` | `3.7e-4` / `3.1e-4` / `8.6e-4` | `3.7e-3` / `3.2e-3` / `9.5e-3` | `2.4e-3` / `2.2e-3` / `5.4e-3` | `6.3e-3` / `6.0e-3` / **`1.38e-2`** | no |
| gauss `5e-4` | `6.1e-4` / `4.9e-4` / `1.4e-3` | `6.2e-3` / `5.3e-3` / `1.6e-2` | `4.0e-3` / `3.6e-3` / `9.0e-3` | `1.1e-2` / `9.9e-3` / `2.3e-2` | no |
| gauss `1e-3` | `1.2e-3` / `9.6e-4` / `2.8e-3` | `1.2e-2` / `1.1e-2` / `3.3e-2` | `8.1e-3` / `7.3e-3` / `1.8e-2` | `2.1e-2` / `2.0e-2` / `4.6e-2` | no |
| gauss `1e-2` | `1.2e-2` / `9.5e-3` / `2.8e-2` | `9.5e-2` / `7.8e-2` / `0.25` | `8.1e-2` / `7.3e-2` / `0.18` | `0.21` / `0.20` / `0.44` | no |
| shots `1e4` (about `2.3e-2` per coordinate) | `2.7e-2` / `2.2e-2` / `6.5e-2` | `0.24` / `0.13` / `0.57` | `0.18` / `0.16` / `0.40` | `0.65` / `0.53` / `1.36` | no |
| shots `1e6` (about `2.3e-3`) | `2.7e-3` / `2.1e-3` / `6.2e-3` | `3.0e-2` / `2.4e-2` / `8.4e-2` | `1.8e-2` / `1.6e-2` / `4.0e-2` | `5.1e-2` / `4.7e-2` / `0.11` | no |
| shots `1e8` (about `2.3e-4`) | `2.7e-4` / `2.3e-4` / `6.4e-4` | `2.9e-3` / `2.4e-3` / `7.4e-3` | `1.8e-3` / `1.5e-3` / `4.0e-3` | `5.1e-3` / `4.8e-3` / **`1.13e-2`** | no (narrowly) |
| shots `2e8` (about `1.6e-4`) | `1.9e-4` / `1.7e-4` / `4.6e-4` | `2.1e-3` / `1.7e-3` / `5.2e-3` | `1.3e-3` / `1.1e-3` / `2.9e-3` | `3.6e-3` / `3.4e-3` / **`7.9e-3`** | **yes** |

**Results.**

- **Among the tested levels.** In the gauss model all four marginal 95th
  percentiles are at most `1e-2` at `sigma = 1e-4` and `2e-4`. They are not
  at `3e-4`, `5e-4`, `1e-3` and `1e-2`.
  - This is a finite comparison of 50-draw empirical percentiles of an
    unbiased, independent Gaussian surrogate.
  - It does not make `2e-4` a necessary or sufficient precision.
  - The percentiles carry about 10 percent sampling uncertainty.
- **What binds the contract.** The energy at `T = 20` does (`p95 = 9.2e-3` at
  `2e-4`). Each readout error changes the energy by about
  `||grad H|| sigma ~ 5 sigma`. These independent errors add over the 21
  readout batches, so the energy error random-walks. Worst-case errors of
  the same size would add linearly.
- **The shots model** (a Gaussian surrogate, section 1).
  - It fails narrowly at `1e8` shots per amplitude and holds at `2e8`.
    Linear interpolation puts the crossing at about `1.3e8` per amplitude,
    which is `1.3e8 x 31 x 21 = 8.5e10` window solves over the run.
  - Plain sampling therefore costs `1/sigma^2`. Amplitude estimation costs
    `pi ||Y_1|| / epsilon_coord` calls per coordinate for the encoded
    increment (section 4): `8.4e3` to `2.7e4` at `||Y_1||` from `0.54` to
    `1.72` and `2e-4`, against `3.7e4` with `||x|| = 2.28` as first
    published.
  - The ledger below assumes amplitude estimation.
- **Scaling with the noise.** Errors scale linearly with `sigma`. Already at
  `sigma = 1e-4` the readout noise sets the error: the 95th percentiles are
  12 to 38 times the noise-free values.

## 3. The re-lift count trade

Order 3, re-lift counts `R = 5, 10, 19, 40` over `t = 0..20` (windows of
length `20/(R+1)`). Each count is run noise-free (one run) and with gauss
noise at `sigma = 2e-4` (50 draws). Receipt: `r3_relift_sigma2e-4.json`.

| `R` (window length) | noise-free: site0 `T=4` / `T=20` | noise-free: energy `T=4` / `T=20` | `sigma = 2e-4`, `p95`: site0 `T=4` / `T=20` | `sigma = 2e-4`, `p95`: energy `T=4` / `T=20` | contract holds: noise-free / noisy | readout batches of 31 coordinates (`R` restarts plus the two outputs) |
|---|---|---|---|---|---|---|
| 5 (3.33) | `4.8e-3` / `1.3e-3` | **`1.57e-2`** / `7.2e-3` | `5.1e-3` / `3.0e-3` | `1.8e-2` / `1.07e-2` | no / no | 7 |
| 10 (1.82) | `1.3e-4` / `1.1e-3` | `4.9e-3` / **`1.13e-2`** | `4.5e-4` / `3.8e-3` | `7.4e-3` / `1.71e-2` | no / no | 12 |
| **19 (1.00)** | `2.6e-5` / `1.6e-4` | `8.4e-5` / `1.2e-4` | `5.8e-4` / `6.3e-3` | `3.6e-3` / `9.2e-3` | **yes / yes** | 21 |
| 40 (0.49) | `4.9e-7` / `9.1e-7` | `3.4e-7` / `2.2e-6` | `8.3e-4` / `8.6e-3` | `5.7e-3` / **`1.10e-2`** | yes / no | 42 |

**Reading.**

- **Too few re-lifts.** With 10 or fewer, the order-3 truncation error alone
  breaks the contract: the energy is off by `1.6e-2` at `T = 4` with
  `R = 5`, and by `1.1e-2` at `T = 20` with `R = 10`.
- **Too many re-lifts.** With 40, the truncation error is negligible (at most
  `2.2e-6`), but the 42 noisy readout batches accumulate. The energy at
  `T = 20` reaches `1.1e-2` at `sigma = 2e-4`, so a smaller `sigma` would be
  needed and every readout would cost more.
- **The window.** Of the four counts tested, only `R = 19` holds the
  contract at `sigma = 2e-4`. Truncation error and accumulated readout
  noise cross between 10 and 40 re-lifts, at most a factor of 4 in count.
  This is a finite comparison, not a unique optimal window or a monotone
  law.

## 4. End-to-end ledger for route (a)

Target T-bare; `R = 19` re-lifts (20 windows of length `Delta t = 1`);
order 3; noise `gauss` at the admissible `sigma = 2e-4` (section 2).
Receipt: `r3_ledger_sigma2e-4.json`.

**Per window.** In window `j` the lifted variable is `u = y / gamma_j`, with
`gamma_j = ||x_r||` at that window's restart. The lifted generator, frozen at
`f_max = 0.8664`, is
`G_j = L_j + f_max M + forcing couplings`, where
`L_j = A_1 + 2 A_2(x_r, .)` and the forcing is
`c_j = F_free(x_r) + f D x_r`.

| quantity | formula | measured, over the 20 windows (min / median / max) |
|---|---|---|
| `gamma_j = ||x_r||` | -- | 2.20 / 2.39 / 2.89 |
| forcing `||F_free(x_r)||`, drive forcing `||D x_r||` | -- | 0.41 / 1.43 / 2.08; 0.40 / 0.53 / 0.69 |
| `||L_j||_2` | -- | 4.61 / 5.08 / 6.22 |
| Carleman dimension, order 3 (Kronecker power) | `31 + 31^2 + 31^3` | 30,783 in every window |
| nonzeros of `G_j` | -- | 614,294 in every window (the origin-centred lift has 310,405; re-centring adds `2 A_2(x_r, .)`) |
| `alpha_BE`, sparse access | `s x max|entry|`, `s = max(row, column) nonzeros`, measured on the frozen `+f_max` generator only; a bound over the driven family `|f| <= f_max` needs both endpoints, since entries are affine in `f` | `122 x 4.0 = 488` in every window |
| `alpha_BE`, LCU bound | `3 (||c_j||/gamma_j + f_max ||D x_r||/gamma_j + ||L_j|| + f_max ||D|| + gamma_j ||A_2||)`, with `||A_2|| = 1.995`; a candidate triangle bound over `|f| <= f_max`, with no circuit | 32.0 / 34.6 / 41.1 |
| first-block weight at window end | `w_j = (||Y_1||/gamma_j) / sqrt(sum_{k<=3} ||Y_k||^2 / gamma_j^{2k})`, Kronecker, `gamma = ||x_r||`, constant block excluded | 0.765 / 0.882 / 0.972 |
| postselection probability | `w_j^2` | 0.586 / 0.778 / 0.945 |
| inhomogeneous norm ratio | `beta_j = Delta t ||b_j|| / ||U(end)||` (the forcing-to-solution ratio a linear-ODE solver started at `u = 0` carries), with `b_j` frozen at `f_max`. The uniform numerator `Delta t (||c_0|| + f_max ||c_1||) / gamma_j` is 1.13 to 1.62 times larger (r4 recount). Neither is bound to a solver theorem | 0.836 / 1.11 / 1.62 (frozen) |
| solver factor, LCU | `S_j = alpha_LCU Delta t beta_j / w_j` | 32.2 / 45.5 / 67.1; sum over windows **928** |
| solver factor, sparse access | `alpha_sparse Delta t beta_j / w_j` | 464 / 640 / 851; sum **12,886** |
| readout per batch (31 coordinates), as first published | `N_read,j = 31 pi gamma_j / sigma` amplitude-estimation calls (the coordinator's `31/sigma` form gives `1.55e5`) | `1.07e6` / `1.16e6` / `1.41e6`; sum over 20 readouts `2.37e7` |
| readout per batch, corrected (review 027 finding 1) | the postselected first block encodes the increment `Y_1,j`, so the idealized leading factor is `31 pi ||Y_1,j|| / epsilon_coord`, with `epsilon_coord` an absolute coordinate tolerance and `||Y_1,j||` known | `||Y_1,j|| / gamma_j` = 0.235 to 0.692; sum over the executed 21 batches `1.20e7` at `epsilon_coord = 2e-4` (r4 recount) |
| window cost | `N_read x S_j` (every amplitude-estimation call re-runs that window's solver) | as published `3.6e7` / -- / `9.0e7` (LCU) |

**Total for route (a):**

```text
as first published:  Q_a = sum_j N_read,j S_j            ~ 1.1e9 (LCU), 1.5e10 (sparse)
corrected encoding, 21 batches, frozen forcing            ~ 5.6e8 (LCU), 7.7e9 (sparse)
the same with the uniform forcing numerator               ~ 6.8e8 (LCU), 9.5e9 (sparse)
```

These are leading-factor proxies, not end-to-end costs. They leave out:

- the growth constant of the time-ordered propagator of the driven window
  generator (not `sup ||exp(G_fmax t)||`), and polylogarithmic factors;
- access and normalization bounds for the whole driven family;
- preparation of each window's forcing state `|b_j>`. The solver starts at
  `u = 0`, and `x_r + Y_1` is formed classically, so no `|x_r + Y_1>` is
  prepared;
- the classical construction of `G_j` from the readout;
- the solver's discretization and precision;
- the norm, sign and confidence costs of a deterministic readout contract.

**The deterministic readout contract, written out** (not measured).

- **The requirement.** Every one of the `M = 21 x 31 = 651` coordinate
  estimates is within `epsilon_coord` simultaneously, with total failure
  probability `delta = 0.05`. A further 21 norm estimates give
  `M' = 672`.
- **Splitting the tolerance.** Split it with `0 < theta < 1`; the split is
  not selected.
  - The amplitude tolerance is `epsilon_a = theta epsilon_coord / ||Y_1||`.
  - The relative norm tolerance is
    `eta = (1 - theta) epsilon_coord / ||Y_1||`.
  - Then `|delta Y_1,i| <= ||Y_1|| (|delta a_i| + eta |a_i| + eta |delta a_i|)`.
- **Calls per signed amplitude.**
  `c_AE (pi / epsilon_a) c_med ln(M'/delta)`, where:
  - `c_AE = 2` is the signed amplitude from a Hadamard-test probability
    under the canonical estimation bound;
  - `c_med = 1/(2 (8/pi^2 - 1/2)^2) = 5.18` is the Hoeffding median boost;
  - `ln(672/0.05) = 9.51`.

  Together that is `98.6 / theta` times the idealized leading factor.
  Other estimators change these constants.
- **The norm**, once per batch, costs `c_AE (pi / eta) c_med ln(M'/delta)`
  calls.
- **The sign.** Each signed-amplitude call is one controlled window solve
  interfered with a reference basis state. The controlled solver and the
  reference are unpriced.
- **Whether `epsilon_coord = 2e-4` meets the contract is not established.**
  Worst-case errors add linearly over the 21 batches. After the pulse `H` is
  a first integral of the closure, so batch `b` leaves a first-order energy
  error `grad H(x_b) . delta x_b`, of size at most
  `epsilon_coord ||grad H(x_b)||_1`. That sum is not in the receipts.

The readout count alone is `1.20e7` window solves corrected (`2.37e7` as
first published).

**Classical comparison, same target (`r3_classical.json`).** DOP853 on the
committed `closed_scalar_rhs` reaches `t = 20` in `0.19 s` (1,322
right-hand-side evaluations), with error `1.4e-9` on the site-0 occupation and
`8.4e-11` on the energy at `T = 20` against T-bare (`rtol 1e-8`). At
`rtol 1e-12` it takes `0.50 s` (3,458 evaluations). This is one same-target
benchmark, not an asymptotic lower bound or a no-speedup theorem.

**On these proxies route (a) shows no speedup at this size.** It reads all
31 coordinates in each of its 21 batches, so it delivers a classical
description of the 31-dimensional state 21 times. Its readout alone,
`1.20e7` window solves, is set against an ODE that a laptop integrates in a
fifth of a second. Route (a) is an encoding demonstration of the archive
equations, not a demonstrated speedup. The per-batch readout grows with the
number of coordinates read. Whether that exceeds the classical integration
cost at larger `L` is not established here.

## 5. Route (b) beside it: direct 12-qubit simulation of the exact dimer

Target T-physical, the exact cutoff-16 dimer (`r3_routeb.json`; exact norms
from revision 2 and G2).

| quantity | formula | value |
|---|---|---|
| qubits | 4 electronic configurations x two 17-level modes (5 qubits each) | 12 (11 dense) |
| `||H_static||` (cutoff 16), `||drive||`, `max|f|` | -- | 24.68, 1, 0.8664 |
| evolution-time-norm product | `T (||H|| + max|f|)` | 102 (`T = 4`), 511 (`T = 20`) |
| readout per observable per time | `pi ||O|| / epsilon` amplitude-estimation calls at `epsilon = 1e-2`. This assigns the whole `1e-2` to readout; a share `epsilon_read` scales the counts by `1e-2 / epsilon_read` | site0 (number operator, norm 1): 314. Energy: `pi x 24.68 / 1e-2 = 7,754`, with `||H||` as normalization (the LCU `alpha_H >= ||H||` is unmeasured) |
| total, a leading-factor proxy | `sum_T (314 + 7,754) x T (||H|| + max|f|)` | `(8,068)(102 + 511)` = `4.9e6` calls |
| omitted from the total | -- | block-encoding normalization `alpha_H`; preparation of the correlated cutoff-16 ground state; coherent simulation error; total failure probability (four estimates, `ln(4/0.05) = 4.38`) |
| cutoff 16 against 20, computed differences at `T = 4` / `20` | two separately prepared finite models; not an infinite-cutoff bound or an implemented route's error | site0 `5.7e-5` / `2.6e-4`; energy `9.6e-5` / `9.6e-5` |
| model | -- | the finite cutoff-16 Hamiltonian, with no closure discrepancy; its distance from the infinite-cutoff model is not bounded here |

**Side by side.**

| | route (a): re-centred windowed lift | route (b): direct simulation |
|---|---|---|
| target | T-bare (the archive equations) | T-physical (the exact dimer) |
| error against its own target, `p95`, `T = 20` | `6.3e-3` (site0), `9.2e-3` (energy), at `sigma = 2e-4` (Gaussian surrogate) | not computed for an implemented route; the cutoff-16-against-20 difference is `2.6e-4` (site0), `9.6e-5` (energy) |
| error against T-physical | between `max(0, Delta - epsilon)` and `Delta + epsilon`, with `Delta` = `0.412` (site0) and `0.0150` (energy) at `T = 20`, and `epsilon` the error against T-bare | not certified: simulation error unpriced, the readout takes the whole allowance |
| register | 30,783-dimensional lift, about 15 qubits plus ancillas, per window | 12 qubits |
| total calls (leading-factor proxies) | `5.6e8` to `6.8e8` (LCU) or `7.7e9` to `9.5e9` (sparse) corrected; `1.1e9` to `1.5e10` as first published | `4.9e6` |
| classical cost of the same output | `0.19 s` | 3.06 s (cutoff-16 exact propagation, record) |

The ratio of the two proxies was about 230 as first published, and is about
110 to 140 (LCU) with the corrected route (a) leading factor. The two omit
different costs, so neither ratio is an established cost comparison. Route
(b) carries no closure discrepancy. Its total error is not certified, and no
route is selected.

## 6. What remains unmeasured

- **Circuits and access.** No state-preparation circuit (for `x_r` or the
  forcing), no block-encoding circuit for `G_j` and no measured `alpha_H`
  for route (b) exist; only norms and nonzero counts are given.
- **Solver constants.** `C_max` per window (`sup_t ||exp(G_j t)||` frozen;
  the driven case needs the time-ordered propagator), the solver's
  polylogarithmic factors and its discretization error are unmeasured.
- **The readout model.** The noise models assume an unbiased, independent,
  Gaussian error per coordinate; `shots` is a Gaussian surrogate with the
  true norm. They omit:
  - the sign recovery of coordinates near zero;
  - the estimate of `||Y_1||`;
  - correlations between coordinates;
  - a confidence union bound over the `21 x 31 = 651` estimates, a factor
    `ln(651/0.05) = 9.47`, or `9.51` with 21 norm estimates.

  The deterministic contract is written out in section 4, not measured.
- **The driven solver.** Not measured:
  - the time-ordered propagator's growth constant;
  - access bounds for the driven family;
  - forcing-state preparation.
- **Statistics.** 50 draws per row; the 95th percentiles carry roughly 10
  percent sampling uncertainty, which matters at the admissible boundary
  (energy at `T = 20`, `9.2e-3` against `1e-2`).
- **Configuration.** One preparation (exact correlated), one parameter set,
  horizons 4 and 20 only. The windowed route was not run toward the escape at
  `t = 141.41`.
- **Route (b).** Pauli or LCU normalization of the boson-encoded Hamiltonian,
  Trotter or qubitization constants, and a gate count are not given.

## 7. Receipts and hashes

All are in `output/local_runs/paper_v_quantum_target_form_20261010/r3/`
(gitignored). Each process's own timer was under 20 minutes; the longest,
the re-lift sweep, took about 9 minutes. The whole increment (preparation,
reference, scoring, report, review and closure) was not timed, so no
whole-increment bound is established.

| receipt | contents | SHA-256 |
|---|---|---|
| `r3_check.json` | batched velocity against the module (`1.2e-16`); noise-free parity with revision 2 | `b227a5437b2f1edb1b07775c2e88269e816b010cf0d84ecd95e7403c43e37535` |
| `r3_noise_gauss.json` | gauss `1e-4`, `1e-3`, `1e-2`, `R = 19`, 50 draws | `facd9377f06530a78df835331a5235e1cb410c35baf56ba9d5665b693269ff26` |
| `r3_refine_gauss.json` | gauss `2e-4`, `3e-4`, `5e-4` | `1afbec12ee4178eef8dd26f5ab905f64479c0b90796ab3d7d3a821e17604d13c` |
| `r3_noise_shots.json` | shots `1e4`, `1e6`, `1e8` | `1c40c258f9baeba2663e07054db521f40bc7245d858d2213972574d34baae8b4` |
| `r3_refine_shots.json` | shots `2e8` | `86e58646942ea8bfbd9a1045fab4087f2bb6bd0f4a2aaefbbd723f8ae46800d9` |
| `r3_relift_sigma2e-4.json` | `R = 5, 10, 19, 40`, noise-free and gauss `2e-4` | `357676e8083b67b7bb5c558736932319b9fc218599aec6c73a1a4196cde2f1a8` |
| `r3_ledger_sigma2e-4.json` | per-window ledger constants and totals | `2cbb0bf98170edb236d07305ddebf54ead735f64f516b3def4233fc7992a3096` |
| `r3_classical.json` | classical integration of the same target | `6dd0557f4e8b16df1ffe1ed40163089231653640e416f01b6fdd6694741231ba` |
| `r3_routeb.json` | route (b): cutoff 16 against 20 errors, the evolution-time-norm product, readout counts | `643219debce8aa1fadec8b3198276cd47a771c9e474f40bc39f5ea5e708572b9` |
| `r3_route_a.py` | the driver, final state | `4b74f02c93d75ed82650ac18d9cf450d070dd07d070a0df555ac3786ef0789dc` |
| `r3_tabulate.py` | table printer, read-only | `0ee1af02cdf7ac0c284b21e6b071b83d9c6de2442fcefab7992abe0662984041` |

The driver was extended after the first runs, but no receipt's computation
changed. The extensions were:

- the `refine` and `classical` modes;
- the route (b) target label;
- the sparse-access totals in `ledger`.

Reproduction (`PYTHONPATH=paper_5/src`, run from the worktree):

```bash
R=output/local_runs/paper_v_quantum_target_form_20261010/r3/r3_route_a.py
$PY $R check
$PY $R noise gauss
$PY $R noise shots
$PY $R refine gauss 2e-4 3e-4 5e-4
$PY $R refine shots 2e8
$PY $R relift 2e-4
$PY $R ledger 2e-4
$PY $R classical
$PY $R routeb
```

Files to edit (next increment): None until the user's decision on the O3
target (revision 2, section 0).
