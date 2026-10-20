# scaling, exact-cost-law: what the exact driven reference costs as a function
# of site count and phonon cutoff (2026-09-20)

Round key `exact-cost-law`. Branch `claude/paper-v-routes-20260914`, worktree
`/Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914`,
HEAD `63852f96`. Nothing here is adopted evidence and no default, module or
committed setting was changed; every measurement is a new diagnostic script
under `output/local_runs/paper_v_scaling_20260920/`. Machine: the replacement
M4 Pro (24 GB), one process at a time under
`pipelines/shell/ram_guard.py --limit-mb 3000`.

**The question this answers.** The archive moment closure carries a few dozen
coordinates whatever the phonon cutoff. The corrected closure's wall to
`t = 20` at two sites is `5,183.6 s` against the exact cutoff-16 driven
reference's `0.7 s` to `t = 4`
(`MATH/paper_facing/paper_V_high_u_gkba/cheap_repair_route_results_20260918.md`
§1, §3.2). The route pays for itself only if the corrected closure's cost grows
slowly with the site count `L` while the exact propagation's grows
exponentially. This round measures and models **only the exact side**; the
corrected closure's `L` dependence is not measured here, and every Paper V
module is dimer-specific (`DimerParameters`,
`paper_5/src/paper5/stability/hubbard_dimer.py:50`, carries no site count), so
nothing was implemented at `L > 2`.

---

## 1. The state dimension and the Hamiltonian's stored nonzeros

The exact reference's Hilbert space is the electron sector times the truncated
phonon sector. `_build_exact_dimer_model`
(`paper_5/src/paper5/stability/exact_reference.py:257`) builds the electron
sector as the spin-up site index tensored with the spin-down site index --- one
electron of each spin, no Hubbard `U` term in the static Hamiltonian --- and one
truncated oscillator of maximum occupation `M` on each site. So for `L` sites
and phonon cutoff `M`:

    state dimension       D(L, M) = L^2 (M + 1)^L

and the static Hamiltonian's stored nonzero count is

    nnz(L, M) = L^2 [(M + 1)^L - 1]              (the diagonal phonon term)
              + 4 L (L - 1) (M + 1)^L            (the chain hopping, both spins)
              + 2 L M (2 L - 1) (M + 1)^(L - 1)  (the electron--phonon coupling)

At two sites `D(2, 16) = 4 * 289 = 1156` and `nnz(2, 16) = 6728`, which are the
committed model's own shape and stored nonzero count. Both closed forms were
checked against **every one of the 47 timed builds** in this round, `L = 2` to
`7` and `M = 2` to `128`: **0 mismatches** (`fit_cost_law.py`). The stored
nonzeros per state dimension run `4.89` to `7.29` over that range and approach
`9 - 6/L` at large `M`.

The `L`-site build used for `L > 2` is
`output/local_runs/paper_v_scaling_20260920/site_scaling.py:build_chain_model`:
the same open-chain hopping for both spins, the same `omega_ph sum_i b_i^dagger
b_i`, the same `g sum_i (n_i,up + n_i,down)(b_i + b_i^dagger)` and the same
drive operator `0.5 (n_first - n_last)`, all at the committed
`DimerParameters` (`lambda = 1.5`, `gamma = 0.5`, drive amplitude 1). **At
`L = 2` it reproduces the committed `_build_exact_dimer_model` exactly**: the
maximum absolute entry difference of the static Hamiltonian and of the drive
operator is `0.0` at cutoffs `2`, `4`, `8` and `16`, with identical stored
nonzero counts (`dimer_check.json`). What differs from the real thing at
`L > 2` is only that no Paper V closure, coordinate chart or scorer exists
there --- the propagation being timed is the same integrator on the same model
structure at a larger site count.

---

## 2. The step count does not depend on the site count or the cutoff

`exact_holstein_driven_trajectory` integrates with DOP853 at `rtol 1e-9`,
`atol 1e-11` and **`max_step = 0.01`**, sampling the declared `h = 0.0025`
grid. Under that declared maximum step the function evaluations are set by the
horizon alone:

| horizon `T` | samples | function evaluations | fitted `1500 T + 2` |
|---|---:|---:|---:|
| `0.5` | 201 | `752` | `752` |
| `4` | 1,601 | `6,017` | `6,002` |
| `20` | 8,001 | `30,002` | `30,002` |

and that count is **identical at every `(L, M)` measured**: `167` at `T = 0.1`
on all 26 grid points from `(2, 4)` to `(6, 3)`, `752` at `T = 0.5` on all
fifteen cutoffs from `M = 2` to `128`. Without the dense sampling grid (two
requested times instead of 1,601) the same horizons take `140` evaluations at
`T = 0.1` and `4,820` at `T = 4`, i.e. `0.80` to `0.84` of the dense-grid
count: the dense output's extra stages are 16 to 20 percent of the
propagation's arithmetic.

Released, the maximum step is worth a factor `3.1` to `9.7` and acquires a weak
cutoff dependence but no site-count dependence. Over `t = 0` to `0.5`
(`step_count_probe.json`):

| `L`, `M` | accepted steps, `max_step = 0.01` | accepted steps, released | released function evaluations |
|---|---:|---:|---:|
| `2, 4` / `2, 8` / `2, 16` / `2, 32` | 50 | 5 / 6 / 5 / 5 | 62 / 74 / 62 / 62 |
| `2, 64` / `2, 128` | 50 | 7 / 14 | 122 / 194 |
| `3, 8` / `3, 16` / `4, 8` | 50 | 5 / 5 / 5 | 62 / 62 / 62 |

So under the declared settings the entire `L` and `M` dependence of the exact
propagation is the **state dimension**, not the step count.

Recorded alongside: the zero-field ground-state energy at two sites is
`-3.7753265534936196` at `M = 16` and `-3.775357388313102` at `M = 32`, `64`
and `128` (`3.08e-5` apart), and `-4.335115686317053` at `L = 3`, `M = 16`.

---

## 3. The measured cost, and the law

Four stages are timed separately: the sparse build, the zero-field ground state
(`eigsh`, `k = 1`, `which = "SA"`, tolerance `1e-12`), the DOP853 propagation,
and the contraction of each sampled wavefunction into the retained moments.

### 3.1 The phonon cutoff at two sites

`cutoff_sweep_t05.json`, `t = 0` to `0.5` on the declared `h = 0.0025` grid
(201 samples), through the committed `_build_exact_dimer_model`,
`_ground_state` and `_contract_exact_driven_trajectory`:

| `M` | `D(2, M)` | `nnz` | ground state (s) | propagation (s) | contraction (s) | total (s) |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 36 | 176 | 0.0008 | 0.0062 | 0.0255 | 0.0370 |
| 4 | 100 | 536 | 0.0012 | 0.0078 | 0.0274 | 0.0407 |
| 6 | 196 | 1,088 | 0.0015 | 0.0095 | 0.0307 | 0.0459 |
| 8 | 324 | 1,832 | 0.0021 | 0.0112 | 0.0335 | 0.0513 |
| 12 | 676 | 3,896 | 0.0034 | 0.0150 | 0.0432 | 0.0662 |
| 16 | 1,156 | 6,728 | 0.0051 | 0.0203 | 0.0548 | 0.0850 |
| 20 | 1,764 | 10,328 | 0.0067 | 0.0250 | 0.0716 | 0.1083 |
| 24 | 2,500 | 14,696 | 0.0091 | 0.0344 | 0.0961 | 0.1445 |
| 32 | 4,356 | 25,736 | 0.0154 | 0.0523 | 0.1415 | 0.2150 |
| 40 | 6,724 | 39,848 | 0.0259 | 0.0713 | 0.1956 | 0.2990 |
| 48 | 9,604 | 57,032 | 0.0365 | 0.0845 | 0.2525 | 0.3807 |
| 64 | 16,900 | 100,616 | 0.0685 | 0.1338 | 0.4706 | 0.6814 |
| 80 | 26,244 | 156,488 | 0.1145 | 0.1992 | 0.7176 | 1.0418 |
| 96 | 37,636 | 224,648 | 0.1893 | 0.3039 | 1.0157 | 1.5259 |
| 128 | 66,564 | 397,832 | 0.5089 | 0.6445 | 1.7682 | 2.9425 |

Over a dimension factor of `1,849` (`M = 2` to `128`) the total wall grows by
`79.5`; over `M = 16` to `128` (dimension factor `57.6`) it grows by `34.6`.
The whole two-site cutoff sweep, fifteen cutoffs to `t = 0.5`, cost under
9 seconds.

Anchors at two sites and cutoff 16 on the declared grid, through the committed
path: **`t = 4` in `0.622 s`** (build `0.005`, ground state `0.005`,
propagation `0.166`, contraction `0.446`) and **`t = 20` in `3.064 s`**
(propagation `0.818`, contraction `2.236`). These are the record's `0.7 s` and
its `0.24 s` for `0.5` time units, measured here at finer grain.

**The contraction, not the propagation, is the larger part of the exact
reference's wall at every point measured**: `0.65` to `0.72` of it at two
sites, and `0.74` and `0.96` at `L = 3` and `L = 4`. It is a reporting choice
--- the sampling grid `h = 0.0025` and the retained-moment operator set --- not
a property of the propagation.

### 3.2 The site count

`site_grid_{a,b,c}.json` (propagation only, `t = 0` to `0.1`) and
`endtoend_*.json` (all four stages, `t = 0` to `4`). The end-to-end `L`-site
runs contract the full `L`-site retained-moment set explicitly; their
contraction arithmetic is the committed contraction's minus the scalar cumulant
subtractions, which at `L = 2`, `M = 16`, `t = 4` makes it `0.2795 s` against
the committed `0.4459 s`, a calibration factor of `1.595` applied to the
`L > 2` contraction figures below.

| `L`, `M` | `D` | `nnz` | ground state (s) | propagation to `t = 4` (s) | contraction to `t = 4`, declared grid (s) | total (s) |
|---|---:|---:|---:|---:|---:|---:|
| `2, 16` (committed path) | 1,156 | 6,728 | 0.0053 | 0.166 | 0.446 | **0.622** |
| `3, 16` | 44,217 | 300,840 | 0.134 | 3.332 | 15.53 | **19.07** |
| `4, 8` | 104,976 | 746,480 | 0.351 | 6.79 | 173.5 | **180.7** |
| `4, 16` (propagation only) | 1,336,336 | 9,747,376 | --- | 78.5 | --- | --- |

The `4, 16` propagation was measured at `62.88 s` for `4,820` function
evaluations without the dense sampling grid and is reported above scaled to the
declared grid's `6,017`. The `4, 8` contraction was measured on 40 samples
(`0.06770 s` per sample) and scaled to the declared grid's 1,601.

Going from two sites to three at cutoff 16 multiplies the exact wall to `t = 4`
by `30.7` (`0.622 s` to `19.07 s`); the state dimension multiplies by `38.2`.

### 3.3 The law

Fitting the measured seconds per right-hand-side evaluation over **all 47 timed
propagations**, `D` from `36` to `1,336,336`, by relative least squares:

    seconds per evaluation = 1.098e-5 + 1.166e-8 * D
      measured / fitted: minimum 0.718, maximum 1.360, median 1.054 (47 points)

Restricted to `D >= 1e4` (21 points) the offset is unnecessary and the fit is
`1.135e-8 * D` (ratio `0.860` to `1.238`, median `1.007`); against the stored
nonzero count instead it is `1.650e-9 * nnz` (ratio `0.811` to `1.306`). The
two forms are equivalent because `nnz / D` only moves between `4.9` and `7.3`
over the whole range, so **dimension is the single variable**.

The contraction's per-observable, per-sample cost fits `5.6e-6 + 5.9e-9 * D` on
the committed two-site contraction over `D = 36` to `66,564` (reproducing
`0.0255`, `0.0548`, `0.4706` and `1.7682 s` to within 1 percent at
`M = 2, 16, 64, 128`), with the number of retained-moment observables at `L`
sites being

    L^3 + 3 L^2 + L    (= 22 at L = 2, the count in `_contract_matrix_state`:
                        L^2 electron density, L coherent phonon, L^2 normal and
                        L^2 anomalous phonon density, L^3 electron--phonon)

**The cost law.** For `L` sites, phonon cutoff `M`, horizon `T` and the
declared reporting grid `h = 0.0025`, with `D = L^2 (M + 1)^L`:

    exact wall (seconds)
      = (1500 T + 2) * (1.098e-5 + 1.166e-8 * D)          [propagation]
      + (T / 0.0025 + 1) * (L^3 + 3 L^2 + L)
            * (5.6e-6 + 5.9e-9 * D)                       [contraction]
      + ground state and build                            [under 0.6 s to D = 1.3e6]

**Validity and uncertainty.** `L = 2` to `7`, `M = 2` to `128`, `D = 36` to
`1.34e6`, `T = 0.1` to `20`, one process on this machine. The propagation term
is good to `0.72` to `1.36` over 47 points. The whole law reproduces the two
end-to-end two-site measurements to `0.93` and `0.95` (`0.58 s` predicted
against `0.622 s` at `t = 4`, `2.92 s` against `3.064 s` at `t = 20`) and the
two `L > 2` end-to-end measurements to `1.44` (over-predicting `19.07 s` as
`27.48 s` at `L = 3`, `M = 16`) and `0.68` (under-predicting `180.7 s` as
`123.5 s` at `L = 4`, `M = 8`): **take a factor `1.5` either way at `L >= 3`**,
the spread being the contraction term's, since the per-observable cost crosses
from arithmetic-bound to memory-bound as the state vector outgrows cache. The
law assumes the declared `max_step = 0.01`; releasing it divides the
propagation term by `3.1` to `9.7` (section 2).

Resident memory measured `827` to `1,309` bytes per state dimension (peak
`1,105 MiB` at `D = 1,336,336`, `1,931 MiB` at `D = 1,476,225`).

---

## 4. What the law says

Predicted wall at the declared reporting grid (`predict_cost.py`,
`predicted_cost.json`); the `L = 2` and `L = 3` cutoff-16 rows and the `L = 4`
cutoff-8 row are the measured ones of section 3.2.

| `L`, `M` | `D` | propagate `t = 4` (s) | contract `t = 4` (s) | total `t = 4` | total `t = 20` | resident |
|---|---:|---:|---:|---:|---:|---:|
| `2, 16` | 1,156 | 0.15 | 0.44 | `0.58 s` | `2.9 s` | --- |
| `2, 32` | 4,356 | 0.37 | 1.10 | `1.5 s` | `7.4 s` | --- |
| `3, 16` | 44,217 | 3.2 | 24.3 | `27 s` | `137 s` | 0.05 GiB |
| `3, 32` | 323,433 | 22.7 | 174.7 | `197 s` | `986 s` | 0.39 GiB |
| `4, 8` | 104,976 | 7.4 | 116.1 | `123 s` | `617 s` | 0.13 GiB |
| `4, 16` | 1,336,336 | 93.6 | 1,465 | `1,559 s` | `7,791 s` | 1.63 GiB |
| `5, 8` | 1,476,225 | 103 | 2,860 | `2,964 s` | `14,812 s` | 1.80 GiB |
| `5, 16` | 35,496,425 | 2,484 | 68,737 | `71,222 s` | `355,933 s` | 43.3 GiB |
| `6, 4` | 562,500 | 39 | 1,756 | `1,796 s` | `8,974 s` | 0.69 GiB |
| `6, 8` | 19,131,876 | 1,339 | 59,640 | `60,979 s` | `304,743 s` | 23.3 GiB |

**At the record's cutoff 16, the exact reference reaches `L = 4` and stops
there on this machine.** `L = 4`, `M = 16`, `t = 4` is `1,559 s` predicted
(`1,000` to `2,300 s` under the law's factor-`1.5` band) at `1.63 GiB`;
`L = 5`, `M = 16` needs `43.3 GiB`, which is `2.4` times the 18 GB aggregate
memory contract of `agent_guidance/shared/memory-budget.md`, before its
predicted `71,222 s` (20 hours) to `t = 4` is reached. Each added site at
cutoff 16 multiplies the state dimension --- and so the wall --- by
`17 ((L+1)/L)^2`: `38.3` from two sites to three, `30.2` to four, `26.6` to
five.

**Largest `(L, M)` in one hour on this machine.** Two bounds. The memory bound
at a 12 GiB working set --- one process inside the 18 GB aggregate, at the
measured `1,309` bytes per state dimension --- is `D <= 9.83e6`. The wall bound
depends on which cost is being bought, because the contraction's operator count
`L^3 + 3 L^2 + L` grows with the site count while the propagation's does not
(`largest_in_one_hour.py`, `largest_in_one_hour.json`):

| `L` | largest `M` to `t = 4`, declared reporting grid | `D` | wall | largest `M` to `t = 4`, propagation only | `D` | wall |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 1,567 | 9.83e6 | `2,732 s` | 1,567 | 9.83e6 | `688 s` |
| 3 | 85 | 5.72e6 | `3,483 s` | 102 | 9.83e6 | `688 s` |
| 4 | 19 | 2.56e6 | `2,985 s` | 27 | 9.83e6 | `688 s` |
| 5 | 8 | 1.48e6 | `2,964 s` | 12 | 9.28e6 | `650 s` |
| 6 | 4 | 5.63e5 | `1,796 s` | 7 | 9.44e6 | `661 s` |
| 7 | 2 | 1.07e5 | `515 s` | 4 | 3.83e6 | `268 s` |
| 8 | 2 | 4.20e5 | `2,860 s` | 3 | 4.19e6 | `294 s` |

Memory binds at `L = 2`, `3` and `4` when only the propagation is bought and at
`L = 2` alone when the declared reporting grid is; the wall binds everywhere
else. To `t = 20` the same table reads `M = 803`, `49`, `13`, `5`, `3`, `2` and
`1` at `L = 2` to `8` on the declared reporting grid, and `M = 1,567`, `102`,
`27`, `12`, `7`, `4` and `3` propagation-only (where the memory bound is again
what binds at `L = 2` to `6`).

So at cutoff 16 an hour buys `t = 4` at up to four sites; `t = 20` at four
sites and cutoff 16 is a `7,791 s` job, and cutoff 16 at five sites is out of
reach of this machine's memory at any horizon.

**What the corrected closure's wall buys on the exact side.** The corrected
closure reaches `t = 20` at two sites and cutoff 16 in `5,183.6 s` at an
all-coordinate scalar RMS of `5.604e-2`. The same `5,183.6 s` spent on the
exact reference at the same horizon and the same declared reporting grid buys
(`break_even.py`) two sites at cutoff `965`, three at cutoff `56`, four at
cutoff `14`, five at cutoff `6`, six at cutoff `3` (`2,364 s`, the next cutoff
exceeding the budget) and seven at cutoff `2` (`2,574 s`). So on today's
measured exact side the closure's wall is already worth four sites at a cutoff
one short of the record's, and the crossing the route needs is a statement
about the corrected closure's own `L` dependence, which is unmeasured.

---

## 5. What this does not settle

- **The corrected closure's `L` dependence is not measured.** Nothing in this
  round touches the archive moment closure, the Dykstra repair, the declared
  proximal log-det completion or the re-selection rule, all of which are
  dimer-specific. The crossing point at which the corrected closure becomes
  cheaper than the exact reference cannot be stated from this side alone.
- **The `L > 2` timings are the same integrator on the same model structure,
  not a Paper V route.** They carry no closure, no chart, no scorer and no
  accuracy figure; the only claim is wall time against state dimension. The
  `L = 2` build is the committed model bitwise (section 1), which is what makes
  the `L > 2` points comparable.
- **The exact reference's cost is dominated by reporting, not propagation.**
  Two thirds to nineteen twentieths of the measured wall is contracting 1,601
  samples into the retained moments. A coarser reporting grid or a smaller
  retained set moves the exact side's cost by that factor without touching the
  physics, so a comparison of route walls is a comparison of two declared
  reporting choices as much as of two propagations.
- **No cutoff-convergence claim.** The ground-state energies of section 2 are
  the only convergence quantity measured; whether cutoff 16 suffices at
  `L = 3` or `4` under the drive is not measured here, and if the required
  cutoff itself grows with `L` the exponent in `(M + 1)^L` is optimistic.
- **The law is a single-process, single-thread measurement on this machine.**
  SciPy's sparse matrix-vector product is not threaded here; a threaded or
  Krylov-exponential propagator would change the constants, not the
  `(M + 1)^L` growth.

## 6. Artifacts

All under
`output/local_runs/paper_v_scaling_20260920/` (git-ignored; nothing committed):

| file | what it holds |
|---|---|
| `cutoff_sweep.py`, `cutoff_sweep_t05.json`, `smoke_t05.json` | the two-site cutoff sweep through the committed path, stage by stage |
| `horizon_t4.json`, `horizon_t20.json` | the two-site cutoff-16 anchors at `t = 4` and `t = 20` |
| `site_scaling.py`, `dimer_check.json` | the `L`-site build and its exact agreement with the committed dimer model at `L = 2` |
| `site_grid_a.json`, `site_grid_b.json`, `site_grid_c.json`, `site_L4M16_t4.json`, `site_L5M8.json` | 26 `(L, M)` propagation timings, `L = 2` to `7` |
| `endtoend_chain.py`, `endtoend_L2_M16_t4.json`, `endtoend_L3_M16_t4.json`, `endtoend_L4_M8_t4.json` | all four stages with the full `L`-site retained-moment contraction |
| `step_count_probe.py`, `step_count_probe.json` | the declared maximum step against the released one |
| `fit_cost_law.py`, `fit_two_terms.py` | the closed-form checks and the fit |
| `predict_cost.py`, `predicted_cost.json` | the law evaluated over the `(L, M)` grid |
| `largest_in_one_hour.py`, `largest_in_one_hour.json` | the one-hour bound under the wall and the 12 GiB working set |
| `break_even.py` | the `(L, M)` the corrected closure's `5,183.6 s` buys on the exact side |
