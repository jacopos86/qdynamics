# The bare closure's instability and its linearized flow

**Round of 2026-10-10, branch `claude/paper-v-routes-20260914`, worktree
`Holstein_test.worktrees/claude-paper-v-routes-20260914`, HEAD `b9f99271`.**
Exploratory local work, not promoted evidence; no default, acceptance rule or
module under `paper_5/src` was changed, nothing is committed, nothing here is
manuscript prose. Exact references enter only as initial conditions of
diagnostic propagations and as reporting inputs. Every script and artifact is
under `output/local_runs/paper_v_instability_20261010/`.

Throughout: "the bare closure" is the unrepaired archive closure,
`closed_scalar_rhs` in the 31 closed scalar coordinates, at `lambda_ep = 1.5`,
`gamma = 0.5`, drive amplitude `1`, phonon cutoff `16`; `J` is its Jacobian
`dF/dx` at a stated state and time; "the leading real part" is the largest
real part of the eigenvalues of `J`; "the `1e4` event" is the first time the
maximum absolute coordinate reaches `1e4`.

## 0. The answer first

1. **Which trajectory "ends near t = 434".** The `1e4` event at
   `t = 433.8964` belongs to the bare closure started from the *exact driven
   trajectory's state at t = 60* (Hartree--Fock preparation), not to one
   started at `t = 0`. The bare closure started at `t = 0` from the *exact
   correlated preparation* fires the `1e4` event at **`t = 141.4057`**, one
   and a half units past the `140`-unit window in which the record reports it
   "never" fires (maximum `24.2421`); that sentence of
   `divergent-initialization-20260920.md` section 2.2 was a window shorter
   than the event time, the same failure section 13 found in the basin test.
   Both trajectories are analysed below; the `t = 60` one is called
   `exact_t60`, the `t = 0` one `exact_correlated_t0`.
2. **It is not numerical.** Under the record's own criterion (working notes,
   "Separating instability from chaos", item 1) every event time converges
   under step halving and quartering, under tolerance tightening and across
   an explicit and an implicit method, at all three starts (table 3.1). The
   record's `433.8964` is the DOP853 `rtol 1e-10` value; tightened, it is
   `434.1811` (`1e-12`) and `434.1878` (`1e-13`), RK4 at `0.00125` gives
   `434.1160`. It is converged to about `0.07`, the size of the shift an
   admissible perturbation of order `1e-11` produces (`7.1e-3` at `1e-12`,
   `1.31` at `1e-10`).
3. **It is a genuine dynamical instability of a volume-preserving flow:
   chaotic motion on an unbounded energy shell that ends in unbounded growth
   of the electron--phonon correlation block.** Measured: `Tr J` vanishes at
   every state (at most `6.1e-13`, against `||J||_2` up to `16.4` in the
   bounded phase and `2.8e4` at the event; random off-trajectory states
   included), so phase-space volume is conserved and the finite-time
   Lyapunov spectrum sums to zero (`-8.9e-8` over 374 units); the Eq. (22)
   energy is conserved to six decimals through the `1e4` event; the largest
   finite-time Lyapunov exponent over the bounded phase is `0.047`
   (Benettin, steps `0.01`/`0.005`/`0.0025` agreeing to three digits); the
   time to the `1e4` event from the exact `t = 60` state spreads over `293`
   to `478` units across admissible `1e-4` perturbations, with no member
   before `293`; past the event the magnitude goes from `1e4` to `1e10` in
   `0.25` to `0.33` time units on three of the four starts, the same at
   `rtol 1e-10` and `1e-12`. It is not shown to be a finite-time
   singularity: past `1e10` to `1e12` the growth slows and the integration
   stalls on the rising oscillation frequency. No attractor: the flow cannot
   contract volume.
4. **The leading unstable direction is not the relative-mode phonon
   population.** Along `exact_t60` the relative-mode population carries a mean
   `0.074` (median `0.022`) of the leading eigenvector's squared norm and is
   above one half at `1.65` percent of samples; the direction that actually
   grows (the leading forward Lyapunov vector) carries `0.045` of it, and sits
   in the anomalous phonon density (`0.355`) and the electron--phonon
   correlation (`0.333`).
5. **The repair does not change the bare closure's linearization; it changes
   where it is taken.** The leading real part along the repaired chain over
   `t = 0` to `20` averages `0.177` against `0.211` along the exact
   trajectory and `0.206` along the bare Hartree--Fock trajectory. What
   separates the repaired chain from the bare trajectory in `t = 10` to `20`
   is the *direction* that grows: along the bare trajectory it migrates into
   the electron--phonon correlation block (share `0.31` to `0.57` from
   `t = 12` to `30`) and grows at a mean `0.11` per unit over `10`--`20` and
   `0.14` to `0.17` over `20`--`40`; along the repaired chain it stays in the
   phonon blocks (correlation share `0.05` to `0.23` from `t = 8`) and grows
   at a mean `0.045` over `10`--`20` and `0.004` to `0.049` over `20`--`40`.
6. **For the fault-tolerant linearization route:** the one-step backward-Euler
   matrix `I - dt J` at `dt = 0.0025` has 2-norm condition number `1.008` to
   `1.038` wherever the maximum absolute coordinate is below `5` (median
   `1.009` to `1.012`); the linearized flow amplifies by `11` to `14` over
   `T = 4`, `17.6` to `51` over `T = 20` and `54` to `3,170` over `T = 60`
   (supremum over all start and end times inside the window), and from the
   exact `t = 60` state by `5.2e4` over `200` units and `2.3e9` over `350`;
   the block system that stacks the backward-Euler steps over
   `[t0, t0 + T]` has condition number `1.1e4` to `1.2e4` at `T = 4`,
   `7.9e4` to `2.0e5` at `T = 20` and `5.2e5` to `1.16e7` at `T = 60`,
   against `2,038`, `10,187` and `30,559` for the same system with `J = 0`
   (table 5.1).

## 1. The trajectories

All bare-closure propagations: `closed_scalar_rhs` (the module function),
DOP853 `rtol 1e-10`, `atol 1e-12`, samples every `0.0025`, terminal `1e4`
event (`trajectories.py`; the batched copy of the vector field used for the
Jacobians reproduces `closed_scalar_rhs` to `6.1e-14` relative at random
states of scale `500` and to `1.0e-15` at stored states of the record's
unrepaired Hartree--Fock trajectory, repaired chain, exact reference and
unrepaired-from-exact-`t = 60` trajectory, `verify_batched.json`).

| name | start | `1e4` event, this round | record | samples |
|---|---|---:|---:|---:|
| `exact_t60` | exact driven trajectory's state at `t = 60`, Hartree--Fock preparation | `433.8964203` | `433.8964` | `149,559` |
| `repaired_t60` | repaired chain's state at `t = 60` (segment 14) | `410.7827012` | `410.7827` | `140,314` |
| `exact_correlated_t0` | exact correlated preparation at `t = 0` | **`141.4057436`** | "never, through `t = 140`" | `56,563` |
| `hartree_fock_t0` | Hartree--Fock/zero-correlation preparation at `t = 0` | `54.4612958` | `54.4613` | `21,785` |
| `repaired_chain` | the stored chain, segments 1 to 14, `t = 0` to `60` | --- | --- | `24,001` |
| `exact_reference` | the exact driven trajectory stored beside it | --- | --- | `24,001` |

The last two are not bare-closure trajectories; the bare closure's Jacobian is
evaluated at their stored states.

## 2. Task (a): the Jacobian along the exact-state bare trajectory

`spectra.py`; the Jacobian is exact (the field is a polynomial of degree two
--- the cubic parts of the normal-particle and normal-hole sources cancel,
as the worklog block of 2026-10-10 on the explicit polynomial form records ---
and the five-point stencil has no truncation error up to degree four),
checked against two-point differences of the module function to `3e-8`
relative. Every stored sample. That block also finds the field exactly
divergence-free; this round measures the same independently (`Tr J` at most
`6.1e-13` along every trajectory, `log |det Phi|` below `3e-10`).

### 2.1 The leading real part

| quantity | `exact_t60` (`t = 60` to `433.90`) | `exact_correlated_t0` (`0` to `141.41`) | `hartree_fock_t0` (`0` to `54.46`) |
|---|---:|---:|---:|
| leading real part at the start | `0.2700` | `0.1571` | `4.5e-16` (spectrum purely imaginary) |
| first time it exceeds `1e-8` | `60.0` (first sample) | `0.0` | `0.0025` (first step, value `0.0015`) |
| fraction of samples above `1e-8` | `1.000` | `1.000` | `0.99995` (all but `t = 0`) |
| first time above `0.5` / `1` / `5` | `378.21` / `428.77` / `433.35` | `20.95` / `138.28` / `140.88` | `27.56` / `39.30` / `53.97` |
| largest, and when | `12.42` at `433.838` | `18.33` at `141.405` | `18.17` at `54.44` |

**When it first becomes positive: immediately.** The leading real part is
positive at every sample of every trajectory except the Hartree--Fock product
state itself at `t = 0`, whose spectrum is purely imaginary (the record's
`2.8e-16` in the 13 coordinates); it is `0.0015` one step later and above
`0.1` by `t = 0.355`. Since `Tr J = 0`, eigenvalues with positive real part
are balanced by eigenvalues with negative real part at every instant (13 to
15 of the 31 are positive at the six times checked).

**Stationary for 270 units, rising slowly to `t = 420`, then a runaway in
the last fourteen.** Ten-unit windows along `exact_t60`:

| window | mean | min | max | maximum `\|x_i\|` |
|---|---:|---:|---:|---:|
| `60`--`70` | `0.179` | `0.005` | `0.320` | `1.556` |
| `100`--`110` | `0.219` | `0.016` | `0.396` | `1.535` |
| `200`--`210` | `0.187` | `0.020` | `0.349` | `1.781` |
| `300`--`310` | `0.208` | `0.038` | `0.364` | `2.316` |
| `400`--`410` | `0.321` | `0.071` | `0.647` | `3.139` |
| `420`--`430` | `0.553` | `0.127` | `1.261` | `5.872` |
| `433`--`434` | `6.431` | --- | `12.416` | `9768.2` |

The ten-unit window means are `0.15` to `0.22` from `t = 60` to `330` with
no trend, and `0.21` to `0.32` from `330` to `420`; inside every window the
leading real part swings from a minimum of `0.005` to `0.07` to a maximum of
`0.32` to `0.65`. The leading real part is therefore a
poor predictor of growth: integrated over `70` to `425.6` it is `78.0`
e-foldings, while a perturbation along the growing direction gains a net
`20.9` over the same window (below and section 3). The eigenvector rotates faster than it grows.

**Realized growth is fluctuating, not burst-dominated.** Along the leading
forward Lyapunov vector (the first column of the QR-reorthonormalized tangent
basis, the direction into which perturbations rotate; `lyapunov_vector.py`),
over the bounded phase `t = 70` to `425.58` (the first time the magnitude
exceeds `5`): mean local exponent `0.059` per unit; `59.8` percent of the
`0.5`-unit intervals are expanding; the net growth `20.86` e-foldings is half
the gross expansion `41.21`; the largest ten percent of intervals carry `38`
percent of the gross expansion. In the last ten units every interval
expands, at mean local exponent `0.47` (`423.5`--`430.5`) and `2.28`
(`430.5`--`433.5`).

### 2.2 The leading unstable eigenvector, and whether it is the repair's direction

Fractions of the eigenvector's squared norm in the lifted Frobenius metric of
`cone_correction.closed_state_lifted_frobenius_metric` (diagonal, so the
blocks partition it). Inside the phonon density the four parts are the
record's: relative-mode population `(N00+N11)/2 - Re N01`, centre-mode
population `(N00+N11)/2 + Re N01`, and the two parts of the relative--centre
coherence `(N00-N11)/2` and `-Im N01` (worklog 2026-09-18), whose squares
with weights `1, 1, 2, 2` partition `||N||_F^2`. Inside the correlation block,
the shared trace `Tr C[q]` (a moment that vanishes on every state) and the
part antisymmetric in the mode index, `(C[0] - C[1])/sqrt 2`, are listed.

| share of the leading eigenvector, mean over all samples | `exact_t60` | `exact_correlated_t0` | `hartree_fock_t0` |
|---|---:|---:|---:|
| `rho`, electronic one-body density matrix | `0.039` | `0.038` | `0.058` |
| `B`, coherent phonon field | `0.149` | `0.068` | `0.054` |
| `N`, phonon density | `0.308` | `0.371` | `0.347` |
| of which relative-mode population | `0.074` (median `0.022`) | `0.095` (median `0.038`) | `0.181` (median `0.105`) |
| of which centre-mode population | `0.058` | `0.072` | `0.072` |
| of which the two coherence parts | `0.089`, `0.088` | `0.096`, `0.107` | `0.050`, `0.043` |
| `A`, anomalous phonon density | `0.211` | `0.256` | `0.345` |
| `C`, electron--phonon correlation | `0.293` | `0.267` | `0.197` |
| of which the shared trace | `0.004` | `0.006` | `0.006` |
| fraction of samples with relative-mode population share above `0.5` | `0.0165` | `0.0334` | `0.1128` |
| cosine of the leading left eigenvector with the relative-mode population's covector, median | `0.078` | `0.131` | `0.088` |

The coherent-field part is entirely in the relative mode (the centre-mode
coherent field is decoupled and never enters the eigenvector). In the final
unit before the event the eigenvector is `0.94` electron--phonon correlation,
and the coordinate that reaches `1e4` is a correlation entry in all four bare
trajectories (`correlation_0_01_real` for `exact_t60`).

**Answer: the unstable direction is not the direction the repair displaces.**
The relative-mode population carries under a tenth of the leading eigenvector
on average and more than half of it at under two percent of samples, and
the leading mode is excited through it with a median cosine of `0.08`. The
direction that actually grows is even less aligned: over the bounded phase
of `exact_t60` the leading forward Lyapunov vector is `0.355` anomalous
phonon density, `0.333` electron--phonon correlation, `0.147` phonon density
and `0.045` relative-mode population, and its median overlap with the plane
of the instantaneous leading eigenvector is `0.048` --- the eigenvector of the
moment is not the direction that grows.

## 3. Task (b): finite-time Lyapunov exponents and the record's criterion

### 3.1 The record's criterion: does the event move under refinement?

| start | RK4 `0.01` | RK4 `0.005` | RK4 `0.0025` | RK4 `0.00125` | DOP853 `1e-8` | DOP853 `1e-10` | DOP853 `1e-12` | DOP853 `1e-13` | Radau `1e-10` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `hartree_fock_t0` | `54.42876` | `54.46091` | `54.46128` | `54.461295` | `54.4612965` | `54.4612958` | `54.46129580` | `54.46129580` | `54.46129580` |
| `exact_correlated_t0` | `141.3129` | `141.4001` | `141.40553` | `141.40586` | `141.39066` | `141.40574` | `141.405883` | `141.405885` | `141.405885` |
| `exact_t60` | `419.507` | `431.519` | `433.973` | `434.116` | `429.806` | `433.896` | `434.181` | `434.188` | `434.472` |

(`steps_*.json`, tolerance pairs `(rtol, atol)` = `(1e-8, 1e-10)` to
`(1e-13, 1e-15)`, the event located inside the RK4 crossing step by cubic
Hermite interpolation.) RK4's successive differences shrink by `4.9` and
`17.2` at `exact_t60` (fourth order is `16`), by `89` and `21` at
`hartree_fock_t0`, by `16` and `16` at `exact_correlated_t0`; DOP853's last
tightening moves the event by `1.8e-10`, `1.2e-6` and `6.7e-3`. **Every event converges; none is a
discretization artifact.** The size of what is left at `exact_t60` is set
by the next table.

**Event-time sensitivity to admissible perturbations** (directions from
`constrained_initial_direction`, tangent to the Eq. (22) energy and the
correlation shared trace, eight seeds per size; DOP853 `1e-10`):

| perturbation size, lifted Frobenius | spread of the event, `hartree_fock_t0` | `exact_correlated_t0` | `exact_t60` |
|---:|---:|---:|---:|
| `1e-14` | `2.7e-11` | `5.1e-8` | `8.1e-5` |
| `1e-12` | `2.2e-9` | `4.4e-6` | `7.1e-3` |
| `1e-10` | `2.2e-7` | `4.4e-4` | `1.31` |
| `1e-8` | `2.2e-5` | `4.4e-2` | `36.1` (`416.2` to `452.3`) |
| `1e-6` | `1.8e-2` | `4.65` | `79.7`, two of eight with no event by `460` |

The spread scales linearly with the perturbation size (spread per unit size
about `2.2e3` from the Hartree--Fock start, `4.4e6` from the exact
correlated start and `7e9` from `exact_t60`, the last saturating near
`1e-10`). The tolerance dependence left at `exact_t60` --- `0.29` between
`rtol 1e-10` and `1e-12`, `0.0067` between `1e-12` and `1e-13`, `0.07`
between RK4 at `0.00125` and DOP853 at `1e-13` --- is the size of the shift
an admissible perturbation of `1e-12` to `1e-10` produces: it is the
trajectory's sensitivity acting on integration error, not an integrator
instability.

### 3.2 Finite-time Lyapunov exponents, existing machinery

`initial_condition_sensitivity.benettin_nearby_trajectory` as written ---
fixed-step RK4, rescaling every `0.5` to `1e-5` in the lifted Frobenius norm,
initial direction `constrained_initial_direction` with the module's seed
`20260803` --- applied to the bare closure with its clock shifted to the start
(`ftle_benettin.py`). Means of the per-interval exponents over the stated
windows:

| start, window | step `0.01` | step `0.005` | step `0.0025` |
|---|---:|---:|---:|
| `exact_t60`, `60`--`100` | `0.0280` | `0.0280` | `0.0280` |
| `exact_t60`, `100`--`200` | `0.0182` | `0.0182` | `0.0182` |
| `exact_t60`, `200`--`300` | `0.0506` | `0.0506` | `0.0506` |
| `exact_t60`, `300`--`400` | `0.0776` | `0.0788` | `0.0790` |
| `exact_t60`, `60`--`400` | `0.0463` | `0.0467` | `0.0468` |
| `exact_t60`, `400`--`419.5` | `1.0170` | `0.0958` | `0.0872` |
| `exact_correlated_t0`, `0`--`141` | `0.1596` | `0.1561` | `0.1558` |
| `exact_correlated_t0`, `20`--`140` | `0.1410` | `0.1400` | `0.1400` |
| `hartree_fock_t0`, `0`--`54` | `0.3062` | `0.3088` | `0.3089` |
| `hartree_fock_t0`, `20`--`54` | `0.4341` | `0.4382` | `0.4383` |

The exponents are step-converged to three digits wherever the three RK4
trajectories are still in their bounded phase; the single outlier,
`1.0170` over `400`--`419.5` at step `0.01`, is the window in which that
step's own trajectory reaches its event (`419.507`). The tangent-linear
counterpart (QR spectrum of the exact Jacobian along the DOP853 trajectory,
`tangent.py`) gives a largest cumulative exponent of `0.153` at `T = 10`,
`0.122` at `20`, `0.084` at `60`, `0.054` at `100`, `0.047` at `140`,
`0.049` at `200`, `0.051` at `300` and `0.093` at the event; its 31
exponents sum to `-8.9e-8`, and an independent DOP853 integration of the
augmented system reproduces its amplification to `1e-8`
(`variational_check_*.json`, `log |det Phi|` below `3e-10`).

### 3.3 The time to the `1e4` event across admissible perturbations

`escape_ensemble.py`: 64 admissible directions of size `1e-4`, DOP853
`1e-10`, `1000` units.

| start | members firing | earliest | quartiles | median | latest | without event at `+300` / `+400` / `+500` |
|---|---:|---:|---:|---:|---:|---|
| `hartree_fock_t0` | `64` | `54.345` | `54.416`, `54.477` | `54.447` | `54.565` | --- |
| `exact_correlated_t0` | `64` | `118.96` | `130.69`, `161.81` | `142.04` | `245.85` | `0`, `0`, `0` (`0.33` at `+150`) |
| `exact_t60`, elapsed from `60` | `64` | `293.1` | `326.1`, `382.3` | `351.7` | `478.1` | `0.97`, `0.17`, `0` |
| `repaired_t60`, elapsed from `60` | `64` | `296.1` | `350.3`, `424.6` | `395.5` | `779.1` | `0.98`, `0.45`, `0.125` |

From the two `t = 60` states no member fires within `293` units and every
member fires by `479` (exact) or `780` (repaired): the time to the event is
not memoryless but has a hard floor, set by a slow growth of the orbit's
amplitude (ten-unit maxima `1.56` at `60`--`70`, `2.36` at `250`--`260`,
`3.14` at `400`--`410` along `exact_t60`) on which the chaotic fluctuation
rides. From the Hartree--Fock preparation the event is deterministic to
`0.22`. This is consistent with the 2026-09-20 basin sweep without any
basin: a single propagation from one boundary is one draw from such a
distribution, and at `t = 60` a `400`-unit window fires with probability
`0.83` from the exact state and `0.55` from the repaired one (the
fractions of members with an event by `+400`), against the sweep's
thirteen of fourteen and seven of fourteen over all boundaries. The
distributions were measured at `t = 60` only; the other thirteen boundaries
were not resampled.

## 4. Task (c): the same Jacobian along the repaired chain, the bare trajectory and the exact one

The repaired chain is propagated by the cheap repair's own vector field; the
Jacobian here is the *bare closure's*, evaluated at the repaired chain's
stored states, which says what the bare closure's linearization looks like
where the repair keeps the state.

### 4.1 The leading real part, `t = 0` to `20` and beyond

| window | bare Hartree--Fock trajectory | repaired chain | exact driven trajectory |
|---|---:|---:|---:|
| `0`--`4` | `0.173` | `0.231` | `0.232` |
| `4`--`6` | `0.181` | `0.123` | `0.216` |
| `6`--`20` | `0.218` | `0.170` | `0.204` |
| `0`--`20` | `0.206` | `0.177` | `0.211` |
| `20`--`40` | `0.381` | `0.139` | `0.212` |
| `40`--`54` | `0.826` | `0.149` | `0.145` |

The bare closure has an unstable mode of the same size at the exact
trajectory's states as anywhere else. Over `0`--`4` the repaired chain's
leading real part equals the exact trajectory's to two digits in every unit
(`0.149`/`0.149`, `0.259`/`0.261`, `0.258`/`0.262`, `0.259`/`0.256`),
because the states agree to the record's `1.08e-2` scalar RMS; over
`4`--`20` it is the lowest of the three in six of nine windows (`4`--`5`,
`5`--`6`, `6`--`8`, then two-unit windows to `20`). The bare
trajectory's leading real part averages the same as the exact one's over
`0`--`20` (`0.206` against `0.211`, window by window above or below it) and
separates only after `t = 20` (`0.381` over `20`--`40`, `0.826` over
`40`--`54`).

### 4.2 What grows, and how fast

Leading forward Lyapunov vector, two-unit windows (the first four units are
the alignment transient of the QR basis and are not read):

| window | local exponent, bare / repaired / exact | correlation-block share, bare / repaired / exact | relative-mode population share, bare / repaired / exact |
|---|---|---|---|
| `4`--`6` | `0.180` / `0.182` / `0.203` | `0.24` / `0.28` / `0.33` | `0.03` / `0.05` / `0.05` |
| `6`--`8` | `0.110` / `0.114` / `0.099` | `0.14` / `0.10` / `0.13` | `0.03` / `0.03` / `0.05` |
| `8`--`10` | `0.026` / `0.037` / `0.041` | `0.15` / `0.10` / `0.11` | `0.02` / `0.14` / `0.24` |
| `10`--`12` | `0.074` / `0.028` / `-0.143` | `0.20` / `0.14` / `0.09` | `0.01` / `0.28` / `0.31` |
| `12`--`14` | `0.188` / `0.031` / `-0.066` | `0.32` / `0.23` / `0.23` | `0.05` / `0.13` / `0.08` |
| `14`--`16` | `-0.029` / `0.037` / `0.162` | `0.49` / `0.07` / `0.16` | `0.04` / `0.20` / `0.21` |
| `16`--`18` | `0.122` / `0.097` / `0.129` | `0.31` / `0.06` / `0.13` | `0.01` / `0.09` / `0.02` |
| `18`--`20` | `0.186` / `0.031` / `-0.144` | `0.31` / `0.09` / `0.41` | `0.02` / `0.09` / `0.16` |
| `20`--`30` | `0.135` / `0.004` / `0.065` | `0.57` / `0.10` / `0.14` | `0.02` / `0.13` / `0.05` |
| `30`--`40` | `0.171` / `0.049` / `0.036` | `0.47` / `0.05` / `0.11` | `0.04` / `0.10` / `0.06` |

Amplification of the bare closure's linearized flow from `t = 0`,
`||Phi(T, 0)||_2`: `13.48` / `13.60` / `13.63` at `T = 4` (bare / repaired /
exact), `47.9` / `26.4` / `15.7` at `T = 20`, and `78.4` / `34.7` for the
repaired chain and the exact trajectory at `T = 60`. Cumulative largest
exponent at `T = 20`: `0.155` / `0.131` / `0.110`; at `T = 60`, repaired
`0.0405` and exact `0.0475`.

**What the repair does to the leading unstable mode in the first six to
twenty units.** It does not remove it, and does not act on it at the linear
level: the leading real part along the repaired chain is the exact
trajectory's to within its window-to-window spread (`0.177` against `0.211`
over `0`--`20`), and the repaired route's velocity departure from the bare
closure (central differences of the stored samples minus the bare field at
the same state) has a median share of `0.000` to `0.022` in the plane of the
leading eigenvector (`route_minus_bare_share_in_leading_mode_plane`; that departure
is `0.989` of its squared norm in the correlation block over `t = 0` to `4`,
`0.892` over `4` to `8`, with the relative-mode population at `0.002` and
`0.050`). What it changes is the state, and with it the direction that grows:
between `t = 10` and `20` the bare trajectory's growing direction moves into
the electron--phonon correlation block (share `0.20` rising to `0.49`, then
`0.57` over `20`--`30`) and grows at a mean `0.11` per unit over `10`--`20`
(`-0.03` to `0.19` by window), which is the block that blows up; the
repaired chain's stays in the phonon blocks (phonon density, anomalous
phonon density and coherent field together `0.66` to `0.91` of it over
`8`--`40`, correlation `0.05` to `0.23`) and grows at a mean `0.045`
(`0.03` to `0.10`). The relative-mode population is a *larger* part of the
growing direction along the repaired chain (`0.09` to `0.28` over `8`--`40`)
than along the bare trajectory (`0.01` to `0.05`), so the repair's
displacement of that coordinate is not what keeps the growing direction out
of the correlation block. The linear amplification over `[0, 20]` is `26.4`
along the repaired chain against `47.9` along the bare one, a factor `1.8`;
the decisive difference is nonlinear (the bare state's maximum absolute
coordinate reaches `5` at `t = 40.77` and `1e4` at `54.46`; the repaired
state's is at most `2.1827` through `60`).

## 5. Task (d): conditioning and amplification for the fault-tolerant linearization route

**Definitions.** `dt = 0.0025` (the cheap repair's declared step and the
stored sample spacing); `J_n` the bare closure's exact Jacobian at the stored
state at `t_n`. (i) `kappa_1`: the 2-norm condition number of the one-step
backward-Euler matrix `I - dt J_n`. (ii) The amplification of the linearized
flow `xi' = J(t) xi` over `[t0, t0 + T]`: `||Phi(t0 + T, t0)||_2`, and the
supremum of `||Phi(t, s)||_2` over all `t0 <= s <= t <= t0 + T` (`s` on a
`0.5` grid, `t` on a `0.05` grid) --- the constant a linear-ODE solver's cost
bound carries for a time-dependent generator. (iii) `kappa_stack`: the 2-norm
condition number of the block lower-bidiagonal matrix that stacks the
`m = T / dt` backward-Euler steps `(I - dt J_n) xi_n - xi_{n-1} = dt s_n`
over the window, the matrix a history-state quantum linear-system solver
inverts; beside it the same quantity with `J = 0` in closed form,
`sin((2m-1)a)/sin(a)`, `a = pi/(2(2m+1))`, which is the part due to the step
count alone. Euclidean norm of the 31 closed scalar coordinates (the lifted
Frobenius metric is diagonal with entries `0.5` to `2`, so it changes any of
these by at most a factor `2`).

### 5.1 Table

| trajectory, `t0` | `kappa_1`, median / max over `T = 60` | amplification `T = 4` | `T = 20`: from `t0` / sup | `T = 60`: from `t0` / sup | `kappa_stack` `T = 4` | `T = 20` | `T = 60` |
|---|---|---:|---:|---:|---:|---:|---:|
| `J = 0` | `1` | `1` | `1` | `1` | `2,038` | `10,187` | `30,559` |
| exact driven trajectory, `0` | `1.0105` / `1.0157` | `13.63` | `15.7` / `17.6` | `34.7` / `54.4` | `12,213` | `78,709` | `522,979` |
| repaired chain, `0` | `1.0087` / `1.0135` | `13.60` | `26.4` / `36.2` | `78.4` / `188.8` | `12,198` | `121,049` | `1,354,749` |
| bare, Hartree--Fock, `0` | `1.0087` / `1.0093` (to `T = 20`) | `13.48` | `47.9` / `49.9` | event at `54.46` | `12,049` | `200,322` | --- |
| bare, exact correlated, `0` | `1.0094` / `1.0123` | `12.98` | `39.3` / `51.0` | `2,457` / `3,169` | `12,071` | `166,189` | `11,558,518` |
| bare, `exact_t60`, `60` | `1.0108` / `1.0130` | `11.44` | `23.4` / `25.2` | `471.7` / `471.7` | `11,248` | `79,396` | `1,448,613` |
| bare, `repaired_t60`, `60` | `1.0092` / `1.0123` | `12.65` | `18.7` / `21.2` | `206.0` / `282.9` | `11,670` | `101,616` | `1,999,948` |

`kappa_stack` is quoted from ARPACK with `sigma_max` to `1e-3` relative and
bracketed above by `max_n ||I - dt J_n|| + 1`; the bracket moves no entry by
more than `0.02` percent. Over the whole of each bare trajectory the median
`kappa_1` is `1.009` to `1.012`; while the maximum absolute coordinate is
below `5` it is at most `1.023` to `1.038`, it first exceeds `1.04` after
that (`t = 431.05`, `408.29`, `138.15` and `40.83` on `exact_t60`,
`repaired_t60`, `exact_correlated_t0`, `hartree_fock_t0`), and it reaches
`3,834` to `5,036` at the event.

**Reading for the quantum-feasibility thread.** The per-step matrix is
trivially conditioned (`kappa_1` at most `1.038`) at the declared step
wherever the closure is bounded, so an implicit-step linear solve costs
nothing through conditioning. The cost sits in the amplification: over `T = 4`
(the pulse) every trajectory amplifies by `11` to `14` and the stacked
system is `5.5` to `6` times worse conditioned than step counting alone; over
`T = 20` the factor is `7.7` (exact trajectory) to `19.7` (bare
Hartree--Fock); over `T = 60` it is `17` (exact trajectory), `44` (repaired
chain), `47` to `65` (bare from the `t = 60` states) and `378` (bare from the
exact correlated preparation, which is `81` units from its event). Longer
windows, measured from the exact `t = 60` state (`||Phi(60 + T, 60)||_2`):
`472` at `T = 60`, `700` at `100`, `2.2e3` at `140`, `5.2e4` at `200`,
`2.3e5` at `250`, `1.3e7` at `300`, `2.3e9` at `350` and `3.7e15` at the
event (`T = 373.9`); past the floor of the time to the `1e4` event (`293`
units from either `t = 60` state across the `1e-4` ensemble) the
linearization is about a trajectory that is itself leaving.

## 6. What kind of instability this is, on the measured evidence

Against the record's own classification table (working notes, "Strange-attractor
hypothesis"):

| observation | measured | supported interpretation |
|---|---|---|
| step, tolerance or method dependence of the failure | none beyond the trajectory's own sensitivity; every event converges (3.1) | not numerical instability |
| initial residual or constraint violation at the start | the Hartree--Fock start is the record's inconsistent preparation; the `t = 60` starts are exact states, and they fail too | not only an inconsistent initial state |
| exponential departure from an unstable fixed point | the leading real part is positive everywhere, but no start is stationary (the Hartree--Fock residual is `0.5303` in the 31 coordinates; the `t = 60` states are mid-trajectory) and the realized exponent (`0.047`) is a quarter of the instantaneous leading real part (`~0.2`) | not a single linearly unstable mode |
| positive largest Lyapunov exponent on a bounded, converged trajectory | `0.047`, step-converged, over `t = 60` to `400` of `exact_t60`, whose maximum absolute coordinate stays below `3.2` until `t = 420`; the bounded phase lasts `293` to `478` units across the `1e-4` ensemble | chaos, transient (it ends in escape) |
| phase-space contraction | none: `Tr J = 0` at every state, exponents sum to `-8.9e-8`, `log \|det Phi\|` below `3e-10`; Eq. (22) energy the same to six decimals at twelve sampled times per bare trajectory, through the `1e4` event (`-3.214773` on `exact_t60`, `-3.556387` on `exact_correlated_t0` after the pulse, `-3.260693` on `hartree_fock_t0` after the pulse) | conservative; no attractor, strange or otherwise |
| unbounded growth with a solver-independent law | `1e4` to `1e10` in `0.253`/`0.257` (exact correlated), `0.325`/`0.321` (repaired `t = 60`) and `0.256`/`0.254` (Hartree--Fock) time units at `rtol 1e-10`/`1e-12`, two-decade intervals shrinking (`0.149`, `0.060`, `0.044`, `0.026` from `1e4` to `1e12`, exact correlated); from `exact_t60`, `1e4` to `1e10` takes `3.0` units at `1e-10` and is not reached in `4.5` at `1e-12` (maximum `2.6e9`); no single power law fits (singular-time-free fits to the running maximum give exponents `6.9` to `8.5` on the exact correlated and repaired `t = 60` starts, log residual `0.05` to `0.42`, against the exponent `1` of a degree-two field's self-similar singularity, and no acceptable fit on the other two, log residual `0.38` to `1.85`); past `1e10` to `1e12` the growth slows and the integration stalls within the 300 s limit (`blowup_*.json`) | converged unbounded growth, i.e. dynamical blow-up, in the electron--phonon correlation block; a finite-time singularity is not established |

**Plainly:** the bare closure's divergence is a genuine dynamical instability
of the closed equations, not a numerical one. The flow is
volume-preserving and conserves the Eq. (22) energy, but that energy is
linear in the phonon density and in the electron--phonon correlation
(`matrix_total_energy`: `omega_ph Tr N` and `4 g Re sum_q C[q]_{qq}`) and
does not contain the off-diagonal correlation entries at all --- the
coordinates that reach `1e4` (`correlation_0_01_real`,
`correlation_1_10_real`, `correlation_0_10_imag`, `correlation_0_01_imag` on
the four bare starts) are exactly such entries. So the energy shell is not
compact and nothing in the closed equations confines those blocks; only the
positivity of the state does, and the closure does not carry it. On that unbounded shell the motion is chaotic with a largest
exponent of about `0.05`, the orbit's amplitude creeps up over hundreds of
units, and when it gets out the correlation block grows without bound, six
decades in about a quarter of a time unit on three of four starts. From the
Hartree--Fock preparation the same growth is reached directly, in `54.46`
units, insensitive to perturbation (spread `0.22` at `1e-4`). The repair does not alter the closure's linearization;
it keeps the state, and with it the growing direction, out of the correlation
block that blows up.

## 7. Reproduction

From the worktree, `PY=/Users/lordskyler/Holstein_test/.venv/bin/python`,
`PYTHONPATH=paper_5/src`, `D=output/local_runs/paper_v_instability_20261010`.
Each process ran under 15 minutes (the longest, RK4 at `0.00125` over 400
units, `116 s`; each blow-up run stops itself at 300 s per tolerance). The
largest process held the `exact_t60` Jacobians, `1.15 GB`; peak resident
was not measured.

```bash
$PY $D/verify_batched.py                       # gate: batched field and Jacobian
/bin/zsh $D/run_trajectories.sh                 # the four bare trajectories (trajectories.py)
/bin/zsh $D/run_spectra.sh                      # Jacobian spectra, every sample (spectra.py)
$PY $D/spectra.py hartree_fock_t0
/bin/zsh $D/run_step_refinement.sh              # 3.1 (step_refinement.py)
/bin/zsh $D/run_ftle_blowup.sh                  # 3.2 Benettin at three steps (ftle_benettin.py);
                                                # its blow-up pass is superseded by run_blowup.sh
/bin/zsh $D/run_tangent.sh                      # amplification, QR spectrum (tangent.py)
$PY $D/tangent.py hartree_fock_t0 --t0 0 --span 54.46
$PY $D/variational_check.py --start hartree_fock_t0 --horizons 4 20
$PY $D/variational_check.py --start exact_t60
/bin/zsh $D/run_lyapvec.sh                      # forward Lyapunov vector (lyapunov_vector.py)
/bin/zsh $D/run_escape.sh                       # 3.3 and 5 (escape_ensemble.py, cond_stacked.py)
/bin/zsh $D/run_blowup.sh                       # blow-up law (blowup_law.py, stepped DOP853)
$PY $D/summarize.py                             # every quoted number -> summary.json
```

`run_conditioning.sh` is the first pass of `cond_stacked.py`, stopped and
replaced (its `J = 0` baseline went through ARPACK on a clustered spectrum;
the closed form is used instead). The directory is `397 MB`, almost all of
it the dense trajectories and spectra.

Artifacts: `traj_*.{json,npz}`, `spectra_*.npz`, `tangent_*.{json,npz}`,
`lyapvec_*.npz`, `benettin_*.{json,npz}`, `steps_*.json`, `escape_*.json`,
`cond_*.json`, `blowup_*.json`, `variational_check_*.json`,
`verify_batched.json`, `summary.json`.
