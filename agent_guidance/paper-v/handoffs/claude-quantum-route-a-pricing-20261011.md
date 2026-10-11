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

## 0. The one sentence

Route (a) meets the `1e-2` contract against the bare closure only if each of
its 20 readouts of all 31 coordinates is accurate to `sigma <= 2e-4`. That
prices it at about `1.1e9` solver calls (LCU access) to reproduce what
classical integration does in `0.2 s`. Route (b) meets the same contract
against the exact dimer in about `4.9e6` calls on 12 qubits.

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
readout at the measurement times `T = 4` and `20`.

- **gauss:** `x_read = x + sigma xi`, with `xi ~ N(0, I_31)`; `sigma` is per
  coordinate, in closed-coordinate units.
- **shots:** each normalized amplitude `a_i = x_i / ||x||` is estimated from
  `S` two-outcome shots (a Hadamard-test estimate, variance
  `(1 - a_i^2) / S`), with `||x||` assumed known. The per-coordinate standard
  deviation is therefore `||x|| sqrt((1 - a_i^2)/S)`, about
  `2.3 / sqrt(S)` at `||x|| = 2.28`.

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
`1e-2` for both observables at both horizons.

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

- **The largest admissible `sigma`.** In the gauss model the 1e-2 contract
  holds at both horizons for `sigma <= 2e-4` per coordinate. It holds at
  `1e-4` and `2e-4` and fails at `3e-4`, `5e-4`, `1e-3` and `1e-2`.
- **What binds the contract.** The energy at `T = 20` does (`p95 = 9.2e-3` at
  `2e-4`). Each restart's readout error changes the energy by about
  `||grad H|| sigma ~ 5 sigma`, and these errors add over the 20 readouts, so
  the energy error random-walks.
- **The shots model.**
  - It fails narrowly at `1e8` shots per amplitude and holds at `2e8`. The
    linear scaling from `1e8` puts the threshold at about `1.3e8` per
    amplitude, which is `1.3e8 x 31 x 20 = 8e10` window solves over the run.
  - Plain sampling therefore costs `1/sigma^2`, against `pi ||x|| / sigma`
    amplitude-estimation calls (about `3.7e4` per coordinate at
    `sigma = 2e-4`).
  - The ledger below assumes amplitude estimation.
- **Scaling with the noise.** Errors scale linearly with `sigma`. Already at
  `sigma = 1e-4` the readout noise sets the error: the 95th percentiles are
  12 to 38 times the noise-free values.

## 3. The re-lift count trade

Order 3, re-lift counts `R = 5, 10, 19, 40` over `t = 0..20` (windows of
length `20/(R+1)`). Each count is run noise-free (one run) and with gauss
noise at `sigma = 2e-4` (50 draws). Receipt: `r3_relift_sigma2e-4.json`.

| `R` (window length) | noise-free: site0 `T=4` / `T=20` | noise-free: energy `T=4` / `T=20` | `sigma = 2e-4`, `p95`: site0 `T=4` / `T=20` | `sigma = 2e-4`, `p95`: energy `T=4` / `T=20` | contract holds: noise-free / noisy | readouts of 31 coordinates (including the final one) |
|---|---|---|---|---|---|---|
| 5 (3.33) | `4.8e-3` / `1.3e-3` | **`1.57e-2`** / `7.2e-3` | `5.1e-3` / `3.0e-3` | `1.8e-2` / `1.07e-2` | no / no | 6 |
| 10 (1.82) | `1.3e-4` / `1.1e-3` | `4.9e-3` / **`1.13e-2`** | `4.5e-4` / `3.8e-3` | `7.4e-3` / `1.71e-2` | no / no | 11 |
| **19 (1.00)** | `2.6e-5` / `1.6e-4` | `8.4e-5` / `1.2e-4` | `5.8e-4` / `6.3e-3` | `3.6e-3` / `9.2e-3` | **yes / yes** | 20 |
| 40 (0.49) | `4.9e-7` / `9.1e-7` | `3.4e-7` / `2.2e-6` | `8.3e-4` / `8.6e-3` | `5.7e-3` / **`1.10e-2`** | yes / no | 41 |

**Reading.**

- **Too few re-lifts.** With 10 or fewer, the order-3 truncation error alone
  breaks the contract: the energy is off by `1.6e-2` at `T = 4` with
  `R = 5`, and by `1.1e-2` at `T = 20` with `R = 10`.
- **Too many re-lifts.** With 40, the truncation error is negligible (at most
  `2.2e-6`), but the 41 noisy readouts accumulate. The energy at `T = 20`
  reaches `1.1e-2` at `sigma = 2e-4`, so the admissible `sigma` falls below
  `2e-4` and every readout costs more.
- **The window.** Of the counts tested, only `R = 19` holds the contract at
  the admissible `sigma`. It is where truncation error and accumulated
  readout noise cross. The window is narrow: between 10 and 40 re-lifts, at
  most a factor of 4 in count.

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
| `alpha_BE`, sparse access | `s x max|entry|`, `s = max(row, column) nonzeros` | `122 x 4.0 = 488` in every window |
| `alpha_BE`, LCU bound | `3 (||c_j||/gamma_j + f_max ||D x_r||/gamma_j + ||L_j|| + f_max ||D|| + gamma_j ||A_2||)`, with `||A_2|| = 1.995` | 32.0 / 34.6 / 41.1 |
| first-block weight at window end | `w_j = (||Y_1||/gamma_j) / sqrt(sum_{k<=3} ||Y_k||^2 / gamma_j^{2k})`, Kronecker, `gamma = ||x_r||`, constant block excluded | 0.765 / 0.882 / 0.972 |
| postselection probability | `w_j^2` | 0.586 / 0.778 / 0.945 |
| inhomogeneous norm ratio | `beta_j = Delta t ||b_j|| / ||U(end)||` (the forcing-to-solution ratio a linear-ODE solver started at `u = 0` carries) | 0.836 / 1.11 / 1.62 |
| solver factor, LCU | `S_j = alpha_LCU Delta t beta_j / w_j` | 32.2 / 45.5 / 67.1; sum over windows **928** |
| solver factor, sparse access | `alpha_sparse Delta t beta_j / w_j` | 464 / 640 / 851; sum **12,886** |
| readout per restart (31 coordinates) | `N_read,j = 31 pi gamma_j / sigma` amplitude-estimation calls (the coordinator's `31/sigma` form gives `1.55e5`) | `1.07e6` / `1.16e6` / `1.41e6`; sum over the 20 readouts `2.37e7` |
| window cost | `N_read,j x S_j` (every amplitude-estimation call re-runs that window's solver) | `3.6e7` / -- / `9.0e7` (LCU) |

**Total for route (a):**

```text
Q_a = sum_j N_read,j * S_j  ~  1.1e9 calls (LCU access), 1.5e10 (sparse access)
```

This leaves out:

- polylogarithmic and `C_max = sup ||exp(G_j t)||` factors (unmeasured);
- state preparation;
- the classical construction of `G_j` from the readout.

The readout count alone is `2.4e7` window solves.

**Classical comparison, same target (`r3_classical.json`).** DOP853 on the
committed `closed_scalar_rhs` reaches `t = 20` in `0.19 s` (1,322
right-hand-side evaluations), with error `1.4e-9` on the site-0 occupation and
`8.4e-11` on the energy at `T = 20` against T-bare (`rtol 1e-8`). At
`rtol 1e-12` it takes `0.50 s` (3,458 evaluations).

**Plainly, route (a) offers no speedup.** It reads all 31 coordinates at
every restart, so it delivers a classical description of the 31-dimensional
state 20 times. Its readout alone, `2.4e7` window solves, prices the
reproduction of an ODE that a laptop integrates in a fifth of a second.
Route (a) is therefore an encoding demonstration of the archive equations,
not a quantum speedup over classical integration of a 31-dimensional ODE.
No asymptotic argument rescues it at this size: the per-restart readout
scales with the number of coordinates, which is the classical cost itself.

## 5. Route (b) beside it: direct 12-qubit simulation of the exact dimer

Target T-physical, the exact cutoff-16 dimer (`r3_routeb.json`; exact norms
from revision 2 and G2).

| quantity | formula | value |
|---|---|---|
| qubits | 4 electronic configurations x two 17-level modes (5 qubits each) | 12 (11 dense) |
| `||H_static||` (cutoff 16), `||drive||`, `max|f|` | -- | 24.68, 1, 0.8664 |
| evolution-time-norm product | `T (||H|| + max|f|)` | 102 (`T = 4`), 511 (`T = 20`) |
| readout per observable per time | `pi ||O|| / epsilon` amplitude-estimation calls at `epsilon = 1e-2` | site0 (number operator, norm 1): 314. Energy: `pi x 24.68 / 1e-2 = 7,754`, with `||H||` as normalization (the LCU `alpha_H >= ||H||` is unmeasured) |
| total | `sum_T (314 + 7,754) x T (||H|| + max|f|)` | `(8,068)(102 + 511)` = **`4.9e6`** calls |
| cutoff error (cutoff 16 against 20), at `T = 4` / `20` | -- | site0 `5.7e-5` / `2.6e-4`; energy `9.6e-5` / `9.6e-5` |
| model error | -- | none: the exact Hamiltonian, no closure |

**Side by side.**

| | route (a): re-centred windowed lift | route (b): direct simulation |
|---|---|---|
| target | T-bare (the archive equations) | T-physical (the exact dimer) |
| error against its own target, `p95`, `T = 20` | `6.3e-3` (site0), `9.2e-3` (energy), at `sigma = 2e-4` | cutoff `2.6e-4` (site0), `9.6e-5` (energy) |
| error against T-physical | adds the closure's `0.41` (site0) and `0.015` (energy) | as above |
| register | 30,783-dimensional lift, about 15 qubits plus ancillas, per window | 12 qubits |
| total calls (leading factors) | `1.1e9` (LCU) to `1.5e10` (sparse) | `4.9e6` |
| classical cost of the same output | `0.19 s` | 3.06 s (cutoff-16 exact propagation, record) |

Route (b) costs about 230 times fewer calls than route (a) even in route (a)'s
cheapest access model, and it has no closure error.

## 6. What remains unmeasured

- **Circuits and access.** No state-preparation circuit (for `x_r` or the
  forcing), no block-encoding circuit for `G_j` and no measured `alpha_H`
  for route (b) exist; only norms and nonzero counts are given.
- **Solver constants.** `C_max = sup_t ||exp(G_j t)||` per window and the
  polylogarithmic factors of the linear-ODE solver are unmeasured, as is the
  solver's own discretization error.
- **The readout model.** The noise models assume an unbiased, independent,
  Gaussian (or binomial) error per coordinate. They omit the sign recovery of
  coordinates near zero, the estimate of `||x||` (assumed known in `shots`),
  correlations between coordinates, and a confidence union bound over the
  `20 x 31 = 620` estimates (the factor `ln(620/0.05) = 9.4`).
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
(gitignored). Each process ran in under 20 minutes; the longest, the re-lift
sweep, took about 9 minutes.

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
