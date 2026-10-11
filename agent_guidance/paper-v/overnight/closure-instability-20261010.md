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

---

# Task 2 (2026-10-10, second round): escape statistics, the predictor, and what confines

**Same branch and worktree, HEAD `a23ff642` (the Task 1 sections above were
committed at `40771435` by the coordinating session); exploratory, not
promoted; no module under `paper_5/src` edited, no default changed, nothing
committed by this round.**
Scripts and outputs in the same directory; the SHA-256 of every artifact a
number below comes from is in `sha256_task2.json` (section 12).

## 8. The answer first

1. **The repaired states escape too, at a rate about five times lower than
   the bare closure's own states, and the advantage over the exact
   trajectory's states shrinks with the time of the state.** Handed to the
   bare closure, 64 admissible `1e-4` perturbations of the repaired chain's
   state reach the `1e4` event after a mean `588.7`, `482.6`, `410.0` time
   units from `t = 20`, `40`, `60` (censored maximum-likelihood rates
   `1.70e-3`, `2.07e-3`, `2.44e-3` per unit); the bare closure's own states
   from the exact correlated preparation at the same times take `136.0`,
   `104.8`, `82.7` (rates `7.35e-3` to `1.21e-2`, `4.3` to `5.0` times
   higher); the exact driven trajectory's states take `128.2`, `291.7`,
   `361.7` (repaired lower by `4.6`, `1.65`, `1.13` times). The time to the
   event is not memoryless: every ensemble has a long delay before its first
   event (no repaired member before `296` to `403` units) and five of
   nineteen are nearly
   deterministic (spread below one unit), so the "rate" is an inverse mean
   time, not a hazard.
2. **Neither representability measure is a leading indicator.** Every member
   loses joint representability long before its event (retained joint Gram
   below `-1` a median `175` to `318` units before), but the time of that
   loss carries no information about when the event comes (rank correlation
   `-0.04` to `0.25`, spread of the lead equal to the spread of the event).
   The crossings that do time the event (rank correlation `0.90` to `0.99`:
   Gram below `-5`, off-diagonal correlation size above `2`,
   Cauchy--Schwarz margin below `-5`) come `9` to `24` units before it ---
   no earlier, and no better, than the maximum absolute coordinate passing
   `4` (`7.7` to `13.7` units, rank correlation `0.92` to `0.99`).
3. **The cone-holding chain's states escape too, and faster as that chain
   heats.** From the task-13 hyperplane-off chain (retained joint Gram held
   to `-1.5e-12` over `t <= 20`, Eq. (22) energy rising from `-3.425` at
   `t = 5` to `-0.622` at `t = 20`), all 64 members of every ensemble reach
   the event: `416.4` units from `t = 5`, then `73.8`, `70.2` and `66.2` from
   `t = 10`, `15`, `20`, against `143` to `156` absolute for the bare closure
   from the same preparation. A representable state at high energy is not
   confined. The same-preparation chain that holds the energy (task 11)
   hands on the longest-lived state measured: from its `t = 20` state 41 of
   64 members have no event in 1500 units (censored rate `2.5e-4`).
4. **Holding the Cauchy--Schwarz bound on the off-diagonal correlations alone
   does not confine.** It is energy-neutral by construction and cheap, it
   lowers the escape rate by `5.2` (Hartree--Fock preparation) and `7.6`
   (exact correlated preparation) times and by `1.3` and `1.0` times from the
   exact and repaired `t = 60` states, and members escape inside 600 units
   from every start (32, 15, 31 and 32 of 32). The escape
   moves to the phonon blocks: the phonon density's diagonal goes negative
   (`N00 = -1.85` at `t = 100`, `-215.9` at `t = 240`), which forces the
   off-diagonal bound to zero, and the anomalous phonon density, the shared
   trace and the diagonal correlation grow until the event at `t = 250.94`.
5. **Holding every second-order Cauchy--Schwarz minor confines.** With the
   phonon density non-negative, the anomalous density, all three Pauli
   components of the correlation and the off-diagonal phonon density each
   bounded by their variances, the Bloch vector inside the unit ball and the
   shared trace pinned at zero, no member of 16 escapes within 600 units
   from any of the four starts, with or without the Eq. (22) energy restored
   (128 member-runs, 76,800 member-units, against controls whose members
   fire at a mean `54.4` to `395.2`). The Eq. (22) energy drifts by up to
   `5.89` without restoration and up to `1.50` with the approximate
   restoration used here; held exactly together with the energy, these
   minors make the reachable set compact (section 11.3), so escape is then
   impossible, not merely slow.

## 9. Task (a): escape statistics

Protocol (`escape_stats.py`): from the stored state at the stated time, 64
perturbations of size `1e-4` in the lifted Frobenius norm along
`constrained_initial_direction` (seeds `1` to `64`; tangent to the Eq. (22)
energy and the correlation shared trace), each propagated by the bare
closure (`closed_scalar_rhs`, DOP853 at `rtol 1e-10`, `atol 1e-12`) to the
`1e4` event or `1500` units (members without an event by then are censored; only one
ensemble has any). Sources
(`state_sources.py`): `bare_exact_start` (the bare closure from the exact
correlated preparation, `traj_exact_correlated_t0.npz`), `bare_hartree_fock`
(from the Hartree--Fock preparation; it has no state at `t = 60`, its event
being at `54.46`), `repaired_chain` and `exact_reference` (the chain's
segment files), `hyperplane_on` (the task-11 base) and `hyperplane_off`
(task 13), both from the exact correlated preparation. "Elapsed" is the time
from the state to the event; "absolute" is the event time; the rates are
the censored maximum-likelihood exponential rate (events over total exposure)
and the same with a delay equal to the earliest event.

| ensemble | fired | mean elapsed | standard deviation | earliest | quartiles and median | latest | mean absolute | rate | rate after the delay |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| bare Hartree--Fock, `t = 20` | 64/64 | `34.45` | `0.04` | `34.35` | `34.4`, `34.5`, `34.5` | `34.5` | `54.4` | `2.90e-2` | --- |
| bare Hartree--Fock, `t = 40` | 64/64 | `14.46` | `0.01` | `14.44` | `14.5`, `14.5`, `14.5` | `14.5` | `54.5` | `6.92e-2` | --- |
| bare exact start, `t = 20` | 64/64 | `135.99` | `33.21` | `113.43` | `120.4`, `123.8`, `134.8` | `296.2` | `156.0` | `7.35e-3` | `4.43e-2` |
| bare exact start, `t = 40` | 64/64 | `104.83` | `18.77` | `90.56` | `97.2`, `101.5`, `105.0` | `216.8` | `144.8` | `9.54e-3` | `7.01e-2` |
| bare exact start, `t = 60` | 64/64 | `82.65` | `2.33` | `80.14` | `81.1`, `81.7`, `84.6` | `90.5` | `142.7` | `1.21e-2` | `3.99e-1` |
| exact reference, `t = 20` | 64/64 | `128.23` | `5.13` | `122.06` | `123.9`, `128.3`, `131.5` | `150.3` | `148.2` | `7.80e-3` | `1.62e-1` |
| exact reference, `t = 40` | 64/64 | `291.69` | `68.00` | `169.22` | `244.4`, `281.0`, `326.8` | `466.8` | `331.7` | `3.43e-3` | `8.17e-3` |
| exact reference, `t = 60` | 64/64 | `361.74` | `45.78` | `293.12` | `326.1`, `351.7`, `382.3` | `478.1` | `421.7` | `2.76e-3` | `1.46e-2` |
| **repaired chain, `t = 20`** | 64/64 | **`588.69`** | `133.20` | `402.88` | `495.1`, `576.7`, `637.3` | `1115.3` | `608.7` | **`1.70e-3`** | `5.38e-3` |
| **repaired chain, `t = 40`** | 64/64 | **`482.65`** | `91.93` | `382.59` | `416.6`, `457.6`, `523.8` | `849.8` | `522.6` | **`2.07e-3`** | `9.99e-3` |
| **repaired chain, `t = 60`** | 64/64 | **`410.04`** | `91.57` | `296.11` | `350.3`, `395.5`, `424.6` | `779.1` | `470.0` | **`2.44e-3`** | `8.78e-3` |
| hyperplane on, `t = 5` | 64/64 | `413.50` | `69.27` | `327.72` | `363.1`, `392.0`, `439.9` | `674.8` | `418.5` | `2.42e-3` | `1.17e-2` |
| hyperplane on, `t = 10` | 64/64 | `199.24` | `0.05` | `199.21` | `199.2`, `199.2`, `199.2` | `199.6` | `209.2` | `5.02e-3` | --- |
| hyperplane on, `t = 15` | 64/64 | `1038.13` | `6.51` | `1028.01` | `1033.8`, `1038.0`, `1040.4` | `1070.9` | `1053.1` | `9.63e-4` | --- |
| hyperplane on, `t = 20` | **23/64** (41 without an event by `1500`) | `1248.67` (fired) | `209.17` | `722.39` | `1073.5`, `1267.0`, `1429.3` (fired) | `1470.5` | `1268.7` (fired) | **`2.55e-4`** | `5.23e-4` |
| hyperplane off, `t = 5` | 64/64 | `416.36` | `20.43` | `385.09` | `402.0`, `411.0`, `428.5` | `474.5` | `421.4` | `2.40e-3` | `3.20e-2` |
| hyperplane off, `t = 10` | 64/64 | `73.80` | `0.24` | `73.40` | `73.6`, `73.7`, `73.9` | `74.9` | `83.8` | `1.36e-2` | --- |
| hyperplane off, `t = 15` | 64/64 | `70.16` | `1.01` | `68.14` | `69.9`, `70.6`, `70.7` | `73.1` | `85.2` | `1.43e-2` | --- |
| hyperplane off, `t = 20` | 64/64 | `66.22` | `0.07` | `66.03` | `66.2`, `66.2`, `66.2` | `66.5` | `86.2` | `1.51e-2` | --- |

(Rates after the delay are omitted where the spread is under one unit: the
event is then effectively deterministic and a rate has no meaning.)

**Is the repaired states' rate lower, by how much, and does it depend on the
time of the state.** Lower than every bare-closure state at every time:
against the bare closure from the exact correlated preparation by `4.3`,
`4.6` and `5.0` times at `t = 20`, `40`, `60`, and against the bare
Hartree--Fock trajectory by `17` and `33` times at `t = 20` and `40` (that
trajectory's own event is at `54.46`, deterministic to `0.1`). Against the
exact driven trajectory's states it is lower by `4.6`, `1.65` and `1.13`
times. It depends on the time of the state: the repaired chain's own rate
rises `1.70e-3`, `2.07e-3`, `2.44e-3` and its mean absolute event time falls
`608.7`, `522.6`, `470.0`, so the later the repaired state, the sooner the
bare closure escapes from it, while the exact trajectory's states move the
other way (`148.2`, `331.7`, `421.7`). The ensembles that start on states
the repair held cones on (task 11, task 13) are erratic in the time of the
state (`413.5`, `199.2`, `1038.1` and more than half beyond `1500` with the
hyperplane on; `416.4`, `73.8`, `70.2`, `66.2` with it off), and five of the
nineteen ensembles have a spread below one unit: from such a state the
event is fixed to within a unit, and the "rate" is the inverse of that time.

**What this says about the repair.** The coordinator's reading holds in the
form "the repair changes the time to escape of the state it hands on, it does
not remove the instability": every repaired state escapes once handed to the
bare closure. On the horizons the repaired chain has actually been run
(`t <= 60`), the earliest escape of any perturbed repaired state is `296`
units after the state, so over `t = 20` to `60` the per-step repair is not
what keeps the chain bounded --- the state at `t = 20` would have survived
more than `400` further units of the bare flow --- which is the same
conclusion the 2026-09-20 boundary sweep reached, now with ensembles.

## 10. Task (b): does the loss of joint representability predict the event?

Along every member, every `0.05` time units (`escape_stats.py`; per-member
values in `escape2_*.json`, every-`0.5` series in `escape2_*_series.npz`):
the minimum unshifted retained joint Gram eigenvalue ("Gram"); the size of
the off-diagonal electron--phonon correlation block,
`sqrt(sum_q |C[q]_01|^2 + |C[q]_10|^2)` ("off-diagonal size"); the smallest
off-diagonal Cauchy--Schwarz margin, `min over q and a in {x, y} of
N_qq (1 - r_a^2) - |Tr(C[q] sigma_a)|^2`, which is the 2-by-2 minor of the
retained joint Gram on the pair `(b_q, sigma_a)` and agrees with the
module's `raw_moment_schur_complement` to `1.1e-16` (`verify_cs.json`)
("CS margin"); and the maximum absolute coordinate. For a threshold, a
member's lead is its event time minus its first crossing.

**As functions of time before the event** (medians over the 64 members;
repaired chain from `t = 20`; the first-50-unit median beside it):

| time before the event | `200` | `100` | `50` | `20` | `10` | `5` | `2` | `1` | `0.5` | `0` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Gram (baseline `-0.047`) | `-0.576` | `-1.12` | `-1.57` | `-2.73` | `-5.33` | `-15.2` | `-71.7` | `-280` | `-1.55e3` | `-2.35e4` |
| off-diagonal size (baseline `0.204`) | `0.504` | `0.802` | `0.925` | `1.30` | `2.13` | `6.37` | `22.8` | `122` | `557` | `1.11e4` |
| CS margin (baseline `0.343`) | `0.293` | `0.060` | `-0.138` | `-1.46` | `-5.13` | `-37.1` | `-477` | `-1.07e4` | `-2.45e5` | `-9.96e7` |
| maximum absolute coordinate (baseline `1.55`) | `1.60` | `1.65` | `1.80` | `2.41` | `3.22` | `6.99` | `27.5` | `99.8` | `384` | `7.1e3` |

All four drift slowly for hundreds of units and then run away in the last
ten to twenty. The exact reference from `t = 40` and the bare closure from
`t = 20` show the same shape (Gram `-0.43`/`-1.63` at `200` before, `-5.0`/
`-7.7` at `10` before; `escape2_analysis.json`).

**Leads at fixed thresholds** --- median lead (interquartile range of the
lead) and Spearman rank correlation of first crossing with event, against
the event's own interquartile range:

| threshold | repaired `t = 20` (event IQR `142.2`) | repaired `t = 40` (`107.2`) | repaired `t = 60` (`74.4`) | exact `t = 40` (`82.4`) | exact `t = 60` (`56.2`) |
|---|---|---|---|---|---|
| Gram `< -1` | `256.8` (`142.0`), `-0.01` | `318.4` (`107.2`), `-0.04` | `177.1` (`70.1`), `0.24` | `174.5` (`78.7`), `0.22` | `207.9` (`56.2`), `0.25` |
| Gram `< -2` | `117.2` (`109.6`), `0.63` | `51.9` (`52.1`), `0.55` | `54.7` (`44.4`), `0.77` | `82.9` (`54.8`), `0.69` | `109.5` (`52.1`), `-0.16` |
| Gram `< -5` | `14.5` (`18.8`), `0.99` | `12.0` (`10.0`), `0.98` | `9.4` (`7.9`), `0.98` | `12.2` (`13.0`), `0.98` | `20.2` (`21.9`), `0.92` |
| off-diagonal size `> 1` | `129.5` (`133.2`), `0.61` | `63.1` (`63.9`), `0.56` | `55.9` (`76.7`), `0.62` | `67.4` (`62.7`), `0.69` | `85.4` (`29.7`), `0.55` |
| off-diagonal size `> 2` | `16.9` (`23.5`), `0.98` | `12.2` (`10.4`), `0.97` | `10.1` (`12.2`), `0.97` | `12.9` (`12.6`), `0.97` | `22.9` (`25.8`), `0.93` |
| CS margin `< -0.1` | `280.0` (`130.9`), `0.12` | `318.3` (`107.2`), --- | `217.0` (`67.9`), `0.42` | `164.2` (`89.6`), `-0.06` | `140.7` (`50.3`), `0.31` |
| CS margin `< -5` | `16.2` (`25.4`), `0.99` | `12.2` (`11.1`), `0.97` | `10.3` (`13.1`), `0.97` | `13.5` (`14.1`), `0.97` | `24.0` (`27.5`), `0.90` |
| maximum absolute coordinate `> 3` | `28.2` (`49.9`), `0.90` | `12.0` (`12.7`), `0.98` | `11.9` (`9.2`), `0.97` | `16.0` (`20.0`), `0.91` | `59.0` (`44.1`), `0.77` |
| maximum absolute coordinate `> 4` | `11.2` (`19.3`), `0.97` | `9.2` (`7.0`), `0.99` | `7.7` (`5.9`), `0.99` | `8.3` (`11.2`), `0.98` | `13.7` (`17.9`), `0.92` |

(`---`: every member crossed at the same sample, so no rank correlation.)
Every member of every ensemble crosses every listed threshold before its
event.

**Reading.** Loss of joint representability always precedes the event and
never times it. The thresholds crossed early (Gram `-1`, CS margin `-0.1`,
which every member passes long before its event)
are crossed at times unrelated to when the event comes: the lead's spread
equals the event's own spread and the rank correlation is `-0.06` to `0.42`
(medians `141` to `318` units before the event).
The thresholds that do time the event (rank correlation `0.90` to `0.99`)
are crossed a median `9` to `24` units before it with a spread of `8` to
`28`, which is what the maximum absolute coordinate does at `4`: the
representability measures are not leading indicators beyond the escape's
own runaway, and are not better indicators than the magnitude. On the
record's standing question this answers the event half: the loss predicts
*that* the bare closure will escape (always, from every state measured), not
*when*. The repaired chain over `t = 0` to `60` never loses the off-diagonal
Cauchy--Schwarz bound at all (margin at least `0` at all 24,001 samples,
`cs_along_chains.json`), nor do the task-11 and task-13 chains over
`t = 0` to `20` (minimum `0.066` and `0.174`); the bare trajectories lose it
at `t = 10.48` (Hartree--Fock preparation, event `54.46`), `4.37` (exact
correlated, `141.41`), `262.0` and `221.2` (from the exact and repaired
`t = 60` states, events `433.90` and `410.78`).

## 11. Task (c): what confines

### 11.1 The cone-holding chain's states, under the bare closure

The task-13 chain holds both cones over `t = 0` to `20` with the Eq. (22)
hyperplane off, and its energy rises. Its states at the ensemble times
(`start_states_table.json`):

| `t` | Eq. (22) energy | retained joint Gram minimum | CS margin | `Tr N` | time to the event under the bare closure (mean of 64) |
|---:|---:|---:|---:|---:|---:|
| `5` | `-3.42519` | `1.2e-15` | `0.221` | `0.988` | `416.36` |
| `10` | `-2.93765` | `-6.3e-16` | `0.584` | `1.810` | `73.80` |
| `15` | `-1.68413` | `-2.3e-15` | `1.036` | `3.205` | `70.16` |
| `20` | `-0.62158` | `+7.5e-3` | `2.407` | `5.314` | `66.22` |

with the same-preparation chain that holds the energy (task 11:
`-3.56578` at every time, Gram `-6.9e-6` to `-3.6e-4`) at `413.50`,
`199.24`, `1038.13` and, at `t = 20`, no event in `1500` units for 41 of 64
members, and the bare closure from the same preparation at absolute `142.7`
to `156.0`. **It escapes, at 64 of
64 members from every one of the four states, and faster the more it has
heated**: from `t = 10` on, within `66` to `74` units. The escape rate is not
zero within 400 units. What this test measures is the state, since from the
handed-on state the flow is the bare closure: a state inside the cone but on
a higher energy shell is not confined. It does not measure the task-13 flow
itself past `t = 20`, which was not run.

### 11.2 Holding a bound in the flow

`clip_flow.py`: the bare closure by RK4 at `h = 0.0025` (it reproduces the
record's bare scores `1.2957e-1`, `2.9372e-1`, `3.3200e-1` at `t = 4`, `8`,
`20` and Eq. (22) RMS `4.517e-2`, `clip_accuracy_hartree_fock_none.json`),
and after every step one map:

- `cs_offdiag` --- the off-diagonal Pauli components of each correlation
  block projected onto the disc their Cauchy--Schwarz bound allows,
  `|Tr(C[q] sigma_a)|^2 <= N_qq (1 - r_a^2)`, `a = x, y`. It touches only
  `C[q]_01` and `C[q]_10`, which Eq. (22) does not contain, so it is
  energy-neutral by construction.
- `minors` --- every second-order bound: shared trace `Tr C[q] = 0` (an
  identity on every state), `|r| <= 1`, `N_qq >= 0`, `|N_01|^2 <= N_00 N_11`,
  `|A_qq|^2 <= N_qq (N_qq + 1)`, `|A_01|^2 <= min(N_00 (N_11 + 1),
  N_11 (N_00 + 1))`, and the Cauchy--Schwarz bound on all three Pauli
  components of each correlation block, each held by radial scaling
  (postconditions verified to `9e-13`, `verify_minors.json`). These are the
  2-by-2 minors of the retained joint Gram that bound each coordinate; the
  full 8-by-8 Gram is not projected.
- `minors_energy` --- the same, then the Eq. (22) energy returned to its
  pre-map value by moving the coherent field along the energy gradient (the
  coherent field enters no centred moment, so no bound is disturbed); when
  the map has raised the energy by more than the coherent field can absorb,
  the restoration stops at the coherent field's energy minimum and the
  residual remains.

**Ensembles** (`1e-4` admissible perturbations, `600` units; 32 members for
`none` and `cs_offdiag`, 16 for the minors):

| start | `none`: fired, mean elapsed | `cs_offdiag`: fired, mean | `minors`: fired | `minors_energy`: fired | energy drift at `600`, `minors` / `minors_energy` |
|---|---|---|---|---|---|
| Hartree--Fock preparation, `t = 0` | 32/32, `54.4` | 32/32, `283.2` (`228` to `345`) | 0/16 | 0/16 | `1.29` / `0.82` |
| exact correlated preparation, `t = 0` | 32/32, `152.7` | 15/32, `481.4` (`411` to `573`) | 0/16 | 0/16 | `5.89` / `0.36` |
| exact reference, `t = 60` | 32/32, `377.4` | 31/32, `468.6` (`387` to `549`) | 0/16 | 0/16 | `1.38` / `1.50` |
| repaired chain, `t = 60` | 31/32, `395.2` | 32/32, `405.1` (`316` to `510`) | 0/16 | 0/16 | `0.055` / `0.0044` |

(The `none` column reproduces the DOP853 ensembles of section 9 and
Task 1: `377.4` against `361.7` and `395.2` against `410.0` from the two
`t = 60` states, median `142.3` against `142.0` from the exact correlated
preparation. Ranges in the `cs_offdiag` column are the 10th to 90th
percentiles of the members that fired. The energy drifts include the pulse's own input, `0.22` and
`0.24` on the two `t = 0` starts. At `600` units the surviving members'
maximum absolute coordinate is `1.41` to `3.60` and their retained joint
Gram minimum `-0.88` to `+0.06`: the minors do not make the full Gram
positive, and do not need to.)

**What escapes when only the off-diagonal bound is held**
(`clip_escape_probe_bare_hartree_fock_t0_cs_offdiag.json`, one unperturbed
member, event at `250.94`): the energy stays at `-3.2606925` throughout;
the phonon density's diagonal goes negative (`N00 = -1.85` at `t = 100`,
`-8.99` at `190`, `-215.9` at `240`), at which point the bound forces the
off-diagonal correlations to zero; the anomalous phonon density
(`4.94` at `100`, `52.2` at `200`, `2.08e3` at `250`), the shared trace
(`0.76`, `6.38`, `312`) and the diagonal correlation (`1.18`, `4.31`,
`231`) carry the growth. Every one of them is held by the `minors` map.

**Accuracy and cost** against the exact driven reference, the record's
all-coordinate scalar RMS of the 31 closed scalar coordinates, Hartree--Fock
preparation (`clip_accuracy_hartree_fock_*.json`):

| map | `t = 4` | `t = 8` | `t = 20` | `t = 40` | `t = 60` | Eq. (22) RMS at `t = 20` | wall per unit, one trajectory |
|---|---:|---:|---:|---:|---:|---:|---:|
| none (bare closure) | `0.12957` | `0.29372` | `0.33200` | `0.45096` | event at `54.46` | `0.04517` | `0.20 s` |
| `cs_offdiag` | `0.12957` | `0.29372` | `0.32442` | `0.38659` | `0.41301` | `0.04517` | `0.21 s` |
| `minors` | `0.13134` | `0.29694` | `0.31450` | `0.34274` | `0.35760` | `0.04802` | `0.46 s` |
| `minors_energy` | `0.13102` | `0.29604` | `0.31360` | `0.33627` | `0.34584` | `0.04733` | `0.42 s` |
| the cheap repair (record) | `0.010803` | `0.037209` | `0.15368` | `0.21606` | `0.25235` | `0.0011888` | `217 s` |

From the exact correlated preparation to `t = 20`: `0.19320` bare,
`0.15739` with `cs_offdiag`, `0.16864` with `minors`, `0.15784` with
`minors_energy` (`clip_accuracy_exact_correlated_*.json`). The maps hold
the electronic one-body density matrix positive from the Hartree--Fock
preparation (minimum `0` to `-2.2e-16` at the sampled times, against
`-216.4` bare) and confine, but they do not make the closure accurate; that
is what the cheap repair's dictionary and extension buy, at about 500 times
the wall.

### 11.3 The quantity that confines

On the evidence, **the confining object is the set of second-order positivity
bounds on every block --- the phonon density non-negative, the anomalous
density, the off-diagonal phonon density and all three Pauli components of
each electron--phonon correlation bounded by the corresponding variances,
the Bloch vector in the unit ball and the correlation's shared trace at
zero --- held together with the Eq. (22) energy.** Each part is measured:
the energy alone does not confine (the bare closure conserves it to six
decimals and escapes from every state measured, section 9); a state inside
the cone does not stay confined once the flow stops holding it (the
task-13 states escape in `66` to `416` units, faster the hotter they are,
11.1); the Cauchy--Schwarz bound on the off-diagonal correlations alone
lowers the escape rate by `1.0` to `7.6` times and confines nothing, because
the energy leaves through the phonon density going negative and the
anomalous density, which Eq. (22) does not contain (11.2); and with every
second-order bound held, no member of 128 escapes in 600 units, with or
without the energy restored. The reason is a short inequality, not a
measurement: with `Tr C = 0`, `|r| <= 1` and `|Tr(C[q] sigma_z)| <=
sqrt(N_qq)`, the Eq. (22) energy is at least
`-2 + omega_ph |B|^2 - 4 sqrt(2) g |B| + omega_ph Tr N - 2 g (sqrt(N_00) + sqrt(N_11))`,
which diverges as `|B|` or `Tr N` does, so a fixed energy bounds `B` and
`Tr N`, and the bounds then bound `A` and every correlation entry: the set
{Eq. (22) energy fixed} intersected with these bounds is compact. Without
the energy it is a cone and is not compact --- the minors-only runs heat by
up to `5.89` in 600 units without escaping, which is consistent with but
does not establish confinement over longer times --- and without the bounds
the energy shell is not compact (sections 6 and 9). The retained joint
Gram, which implies every one of these bounds, is therefore sufficient
with the energy but more than is needed, and the 36-row positive extension
is not needed for confinement at all on this evidence (it was not tested
in isolation; every map above confines without it). **The Cauchy--Schwarz
bound on the off-diagonal correlations alone is not the confining object.**
The ensembles point the same way without any map: the task-11 chain,
which holds the energy and keeps the retained joint Gram within `-3.6e-4`,
hands on at `t = 20` the longest-lived state of all (41 of 64 members
without an event in 1500 units), and the task-13 chain, which holds the
Gram exactly and lets the energy rise, hands on the shortest-lived (`66` to
`74` units from `t >= 10`). What the confining set costs to hold by itself:
a handful of scalar comparisons and about twenty square roots per state per
step --- no eigenvalue
problem, no solve --- measured at `0.42` to `0.46 s` per time unit for one
trajectory in this unoptimised Python, about twice the bare closure's own
step and about one five-hundredth of the cheap repair's `217 s` per unit;
the measured price is in accuracy, `0.314` at `t = 20` against the cheap
repair's `0.154`, and in the energy, which radial scaling does not hold
exactly (drift `0.0044` to `1.50` over 600 units with the restoration
through the coherent field, which fails whenever the map has raised the
energy beyond the coherent field's minimum).

## 12. Reproduction and artifact hashes

```bash
$PY $D/verify_cs.py                    # the CS margin against the module's Gram
$PY $D/verify_minors.py                # energy formula and the minors map
/bin/zsh $D/run_escape2.sh             # section 9 and 10 ensembles (escape_stats.py)
$PY $D/analyze_escape2.py              # -> escape2_analysis.json
$PY $D/cs_along_chains.py              # -> cs_along_chains.json
$PY $D/start_states_table.py           # -> start_states_table.json
/bin/zsh $D/run_clip_accuracy.sh       # 11.2 accuracy (none, cs_offdiag, cs_all)
$PY $D/clip_flow.py accuracy --variant minors --preparation hartree_fock --span 60 --horizons 4 8 20 28 40 55 60
$PY $D/clip_flow.py accuracy --variant minors_energy --preparation hartree_fock --span 60 --horizons 4 8 20 28 40 55 60
$PY $D/clip_flow.py accuracy --variant minors --preparation exact_correlated --span 20 --horizons 4 8 12 16 20
$PY $D/clip_flow.py accuracy --variant minors_energy --preparation exact_correlated --span 20 --horizons 4 8 12 16 20
/bin/zsh $D/run_clip_ensembles.sh      # 11.2, 32 members, 600 units (none, cs_offdiag)
/bin/zsh $D/run_minors_ensembles.sh    # 11.2, 16 members, 600 units (minors_energy, minors)
$PY $D/clip_escape_probe.py --variant cs_offdiag --span 400
$PY $D/sha256_task2.py                 # -> sha256_task2.json
```

Every process ran under 20 minutes (the longest, the `minors` ensembles,
`682` to `687 s`; the 1000-unit clip ensembles first launched were stopped
under machine load and rerun at 600 units with 32 members). The machine was
shared with another agent's runs throughout, so walls are upper bounds.

SHA-256 of the artifacts the numbers come from: the input states and every
JSON output, below; the full list, 137 files including the per-member series,
the `resume.npz` files used to stitch the chains and the scripts, is in
`sha256_task2.json` (rendered by `sha256_table.py`). Hashes were taken after
every run had finished; any later rerun changes them.

| artifact | SHA-256 |
|---|---|
| `output/local_runs/paper_v_instability_20261010/traj_exact_correlated_t0.npz` | `9f48c778280aacced3524e586ac30e519c601411db1c1085d9942f3f67f2f91d` |
| `output/local_runs/paper_v_instability_20261010/traj_hartree_fock_t0.npz` | `5f4e367ebf4b4ff771d51210a0d6906718102c35cbf9d3de16ddf0a37cef61d3` |
| `output/local_runs/paper_v_divergence_hf_dyk120_20260920_seg1/trajectory.npz` | `9f7eefcf015b8d21929b516ac75a58a759f1dd3ef8fd0157054d96f4a82b8deb` |
| `output/local_runs/paper_v_divergence_hf_dyk120_20260920_seg10/trajectory.npz` | `d6752cd7c5db4dfba5472a820190e687bd544ef4b6bac2b8d4c516b5bd61bc40` |
| `output/local_runs/paper_v_divergence_hf_dyk120_20260920_seg11/trajectory.npz` | `ecf20a6e3618278a2bf953e6065049fe8c48b641f966679f0926127fdf21ca9e` |
| `output/local_runs/paper_v_divergence_hf_dyk120_20260920_seg12/trajectory.npz` | `2c534d515b00dd4d941e9d8dcedba3999dd3384a3017c07efcb9637cbb9d6fce` |
| `output/local_runs/paper_v_divergence_hf_dyk120_20260920_seg13/trajectory.npz` | `3da423deb5291333345ef573b0920aa54b3468898b26cde2a4b4be8b0639fc6c` |
| `output/local_runs/paper_v_divergence_hf_dyk120_20260920_seg14/trajectory.npz` | `b9f7f2cc847c4ee5c2315eb6e6167bc97e2e7ceb0b051cdbf4e7ed06c5cbf112` |
| `output/local_runs/paper_v_divergence_hf_dyk120_20260920_seg2/trajectory.npz` | `5b83103d03566a1094077921ae6db02bc15f7b7d2a2b02ba3435d3c324611062` |
| `output/local_runs/paper_v_divergence_hf_dyk120_20260920_seg3/trajectory.npz` | `a64012362e1be6024dd5044ce29af5aaf1707a069e1bab9f016813765bb15780` |
| `output/local_runs/paper_v_divergence_hf_dyk120_20260920_seg4/trajectory.npz` | `6af709eccb04d2dfbd27f9f646c50566ae345bc978649cff2953e496e400c62d` |
| `output/local_runs/paper_v_divergence_hf_dyk120_20260920_seg5/trajectory.npz` | `ec1d0c3feb73f24e44b9bc7b0efd27a690e4a7e6db0fb2c785048785489149f9` |
| `output/local_runs/paper_v_divergence_hf_dyk120_20260920_seg6/trajectory.npz` | `4a1949cb21fed530294f1985fed0e24ea19b1f3ed6624e243ad88131c7157f4d` |
| `output/local_runs/paper_v_divergence_hf_dyk120_20260920_seg7/trajectory.npz` | `ae358252e88d22e8d2f6838746452b419a9e5769088e6e6153a8e18bcc937f47` |
| `output/local_runs/paper_v_divergence_hf_dyk120_20260920_seg8/trajectory.npz` | `dc0dfa74b605a86d9470f19982d8122b26eadbce7aeabc253b60ea55860171b2` |
| `output/local_runs/paper_v_divergence_hf_dyk120_20260920_seg9/trajectory.npz` | `1f6cbd76437763ccd60c0e4a628a6665dd67f5b30fd47d3528403f438adf90ac` |
| `output/local_runs/paper_v_dictionary_energy_none_27key_n16_measured_dyk120_base_20260918o_seg1/trajectory.npz` | `529f40fdcc17c3f40a3233be5f9027a3d4a18ba6fe6a6cb83224ffa12bcb950e` |
| `output/local_runs/paper_v_dictionary_energy_none_27key_n16_measured_dyk120_base_20260918o_seg2/trajectory.npz` | `5833fd381957d3cf58dec0597a3ddee4f82bc671d079c2c139d152aa253595e5` |
| `output/local_runs/paper_v_dictionary_energy_none_27key_n16_measured_dyk120_base_20260918o_seg3/trajectory.npz` | `73744618128e79ea6687959dfc5d9502887344c909d3ca09d3d67c3087b33aa6` |
| `output/local_runs/paper_v_dictionary_energy_none_27key_n16_measured_dyk120_base_20260918o_seg4/trajectory.npz` | `04a74375ce2ba02df2e9e8539d12de6d04e0ecc3cfc8b06f552bc5902fd927bc` |
| `output/local_runs/paper_v_dictionary_energy_power_27key_n16_measured_dyk120_base_20260918m_seg1/trajectory.npz` | `40637207645cf4079b0510e08c7e316ee48a6d8c7e1b96c734e54cb1f76e9ed0` |
| `output/local_runs/paper_v_dictionary_energy_power_27key_n16_measured_dyk120_base_20260918m_seg2/trajectory.npz` | `8598f745ca802f77c6c9918d12a86ce2c075d19d25a2d91c2c9d11d6d3e1ef32` |
| `output/local_runs/paper_v_dictionary_energy_power_27key_n16_measured_dyk120_base_20260918m_seg3/trajectory.npz` | `dc4732197462f65b129c010fa1dfbd4c5f166e718a7a62b2278857f5a0fc0910` |
| `output/local_runs/paper_v_dictionary_energy_power_27key_n16_measured_dyk120_base_20260918m_seg4/trajectory.npz` | `050a76c3cf4dcac6c2dc0a406167cbb48f758da1fad47865c494d8b78c79a110` |
| `output/local_runs/paper_v_dictionary_energy_power_27key_n16_measured_dyk120_base_20260918m_seg5/trajectory.npz` | `b9a1d30c6e49e46a24115f4d469f3b92bd3c9e14dee1508ff21bcadad7c1f2c2` |
| `output/local_runs/paper_v_dictionary_energy_power_27key_n16_measured_dyk120_base_20260918m_seg6/trajectory.npz` | `18be81b3e0f51ac4332d70e22272a1872984d4fb72c1becfdf32072836ef8252` |
| `output/local_runs/paper_v_instability_20261010/escape2_analysis.json` | `5cd7f59df8c4d62894f0239f2a5cfb40cb7aa1b14c01e7243963eb3a4ca588b5` |
| `output/local_runs/paper_v_instability_20261010/escape2_bare_exact_start_t20.json` | `8a642b45c55b1446ec0959ca2939465647a89eb36afab1919e1d77ed7be46498` |
| `output/local_runs/paper_v_instability_20261010/escape2_bare_exact_start_t40.json` | `94b0fcba8c27a65cd8d9c9ef59873a4b28bb9bc47ccf75d3f4e245bc039adda5` |
| `output/local_runs/paper_v_instability_20261010/escape2_bare_exact_start_t60.json` | `a24ce623bf979b9023a24c3747cdc639ddee7e0ea391006ea4f0c736527ba4fd` |
| `output/local_runs/paper_v_instability_20261010/escape2_bare_hartree_fock_t20.json` | `79a100ed1448d0b74df7cd4ce1168165d20dac4d689bd4805c8f2069bdd94a91` |
| `output/local_runs/paper_v_instability_20261010/escape2_bare_hartree_fock_t40.json` | `0e1743d71c0eba7ce735120e09ec2f227145941e9e0e0d04753babdd4600bc10` |
| `output/local_runs/paper_v_instability_20261010/escape2_exact_reference_t20.json` | `f8d95826ac90784e9fbc67e8c36178846851daa86cbbb62aee41512b01820cef` |
| `output/local_runs/paper_v_instability_20261010/escape2_exact_reference_t40.json` | `c3744bd064598bdf4b46f5417e77a6bda107e0ed927d25463f5ffd6325f12fc4` |
| `output/local_runs/paper_v_instability_20261010/escape2_exact_reference_t60.json` | `2e1d19260caccdd6d9a7918f446064515ee64e6f14e0db4b62389d2954e465ef` |
| `output/local_runs/paper_v_instability_20261010/escape2_hyperplane_off_t10.json` | `24f2921eccbe045d027f6c413025d1eceecb916118463e84316913d5b09a4710` |
| `output/local_runs/paper_v_instability_20261010/escape2_hyperplane_off_t15.json` | `ffa989678c778c5e890f15fead5854b661e54b04e5132ba37ee82628cfe293d6` |
| `output/local_runs/paper_v_instability_20261010/escape2_hyperplane_off_t20.json` | `69bb8f62bfeab00c7b7cafcdfb82d32f2d972b53f137993b7ce95ebfb4e83db4` |
| `output/local_runs/paper_v_instability_20261010/escape2_hyperplane_off_t5.json` | `0580596c9d113c3381dedf291b08fa2272354ac9d38be1e79b17c3173570282e` |
| `output/local_runs/paper_v_instability_20261010/escape2_hyperplane_on_t10.json` | `1cffbe6c84bbbab347ab7fc7c625ea4833642dcae283d56cb52afe07b21a1854` |
| `output/local_runs/paper_v_instability_20261010/escape2_hyperplane_on_t15.json` | `9ee7999e4defc33516d5e5911999de83680950b7fb6b3cdf163ecad9d802cc62` |
| `output/local_runs/paper_v_instability_20261010/escape2_hyperplane_on_t20.json` | `262a51bc866f27d93fde4f60d4cc7c7f6586050151f9d144b0012e144ba3a897` |
| `output/local_runs/paper_v_instability_20261010/escape2_hyperplane_on_t5.json` | `f3e62746f90be3dba174008f550d9619363bbcac0e26502f126d461eff35988c` |
| `output/local_runs/paper_v_instability_20261010/escape2_repaired_chain_t20.json` | `f2ea79306bce5a3e093cbe18e778268bb29775722c42903193e4ceba99e03064` |
| `output/local_runs/paper_v_instability_20261010/escape2_repaired_chain_t40.json` | `18e50dd01ab4ed8b925f087b0bed0773c935ab3a77bb19ea83ca5e4fb709f0e6` |
| `output/local_runs/paper_v_instability_20261010/escape2_repaired_chain_t60.json` | `20a969324af2d3bd213e48eb5872f871befa5333c5908c2edd0d241c50c288bb` |
| `output/local_runs/paper_v_instability_20261010/clip_accuracy_exact_correlated_cs_all.json` | `783f7559e9ab7472b0b48d34501f1c9c7e2aa7a29bad48f866fdd96572c98165` |
| `output/local_runs/paper_v_instability_20261010/clip_accuracy_exact_correlated_cs_offdiag.json` | `c5ef066065c8ed489daea2d10735549e00b82d76fed426a45804f9a951d262a0` |
| `output/local_runs/paper_v_instability_20261010/clip_accuracy_exact_correlated_minors.json` | `279bd811c2db8e76d5c2ec7dad3315e7c3d8fbb42cce9415a9810887dbc44a73` |
| `output/local_runs/paper_v_instability_20261010/clip_accuracy_exact_correlated_minors_energy.json` | `75fc7eb6a2da4fe597f3c50423afdecab26bd23b1faeefc27d93ecedc9bff3c4` |
| `output/local_runs/paper_v_instability_20261010/clip_accuracy_exact_correlated_none.json` | `40143125091196f1b606c9d959cc10ae811c925f7cdf8e9d03c95c60afa82848` |
| `output/local_runs/paper_v_instability_20261010/clip_accuracy_hartree_fock_cs_all.json` | `508257d3506559414b28c8842fe848251ea11ce652787c079490104d6fd8d12a` |
| `output/local_runs/paper_v_instability_20261010/clip_accuracy_hartree_fock_cs_offdiag.json` | `0883d58e0c8d0c1f8f9f7d564e01175a378f9388a94e807bd749f536eb2c7c38` |
| `output/local_runs/paper_v_instability_20261010/clip_accuracy_hartree_fock_minors.json` | `38fb303a6292529cfc4b2b2b227ab070c434798b0839d6765d1b67f7f2cb9a8d` |
| `output/local_runs/paper_v_instability_20261010/clip_accuracy_hartree_fock_minors_energy.json` | `9f298948a54543f31446b5139614c3b11eb080e7dc2f3762c17926a8253ac0a3` |
| `output/local_runs/paper_v_instability_20261010/clip_accuracy_hartree_fock_none.json` | `df2a30a9a0bf9949c5793501c4b7a616e90887c992b480d06ef572b47ba41267` |
| `output/local_runs/paper_v_instability_20261010/clip_ensemble_bare_exact_start_t0_cs_offdiag.json` | `c71c81e7ba3a0e802324520a6d3b559c0a8b41f92f6840028e1e0cd26188bfb6` |
| `output/local_runs/paper_v_instability_20261010/clip_ensemble_bare_exact_start_t0_minors.json` | `d3465638ea2f3ff3d4a9f7ecc0cc31bcf30680c53e44f47819af02f5442053e8` |
| `output/local_runs/paper_v_instability_20261010/clip_ensemble_bare_exact_start_t0_minors_energy.json` | `2ea34732fda6aca70b000a5e37c01df813cda70d8f3471b74ba304c44f79284d` |
| `output/local_runs/paper_v_instability_20261010/clip_ensemble_bare_exact_start_t0_none.json` | `5e045148b61d7079ed17a5d7cc77b921c76d265f32aef3a6f72431ee4abd62af` |
| `output/local_runs/paper_v_instability_20261010/clip_ensemble_bare_hartree_fock_t0_cs_offdiag.json` | `d86f9025ecb1d67a532dc7c46de1ca41c791431078c58d20dc3246a4bf62a192` |
| `output/local_runs/paper_v_instability_20261010/clip_ensemble_bare_hartree_fock_t0_minors.json` | `48954b79b4cd2a02a714d3af20ab26914eb7f0d6ae5b8e3ee47ee2055e78b312` |
| `output/local_runs/paper_v_instability_20261010/clip_ensemble_bare_hartree_fock_t0_minors_energy.json` | `0b3ace35aab3ed1952cda3984d4f10cdcaa1aa8759828124b8786941c046c45c` |
| `output/local_runs/paper_v_instability_20261010/clip_ensemble_bare_hartree_fock_t0_none.json` | `e6b1f7a22567425b55ff4a1f40c0ca17017da191b8586cdcfbacdf21c58c9ef1` |
| `output/local_runs/paper_v_instability_20261010/clip_ensemble_exact_reference_t60_cs_offdiag.json` | `7eb6c501a3bade47e9fee5652ce519cf2676cbdfd68da7df7aafe3aaa828be4b` |
| `output/local_runs/paper_v_instability_20261010/clip_ensemble_exact_reference_t60_minors.json` | `a35ab8d3d4058f3f012345e7d882b9077ab43c267af54c735d558c3d72ffad55` |
| `output/local_runs/paper_v_instability_20261010/clip_ensemble_exact_reference_t60_minors_energy.json` | `21cda8b69d581410301c35a39743769f0462fcf5b997b162da275591acc4c8bc` |
| `output/local_runs/paper_v_instability_20261010/clip_ensemble_exact_reference_t60_none.json` | `e8c494fedf9e56da1c7cb945516170fdb1c584a51773e755852e0edb8df39420` |
| `output/local_runs/paper_v_instability_20261010/clip_ensemble_repaired_chain_t60_cs_offdiag.json` | `9627699ae798af4c822ea3aeb8f1e7acda5da1289a20ca7964437fc921f9af36` |
| `output/local_runs/paper_v_instability_20261010/clip_ensemble_repaired_chain_t60_minors.json` | `184e00cc971a3b4e3d035321ca15e63be7952d59824395ecdc2d4168fd57546e` |
| `output/local_runs/paper_v_instability_20261010/clip_ensemble_repaired_chain_t60_minors_energy.json` | `f34517ba5507d995beb50d9b908b1321af19174d66ca4e188fe32465af138dbd` |
| `output/local_runs/paper_v_instability_20261010/clip_ensemble_repaired_chain_t60_none.json` | `ef4bb13535377541949f3a86bd625b71ff247375f1038f2d5e63508f3dcf35fa` |
| `output/local_runs/paper_v_instability_20261010/clip_escape_probe_bare_hartree_fock_t0_cs_offdiag.json` | `a75e18e5d17e42602ee22a9ddb6498aee7a2dfea65594968afff56b741efa446` |
| `output/local_runs/paper_v_instability_20261010/cs_along_chains.json` | `8f162a3da1f4335f32b0e49d6ee93a49b90b2e5cc0a5357fc1b121791bf435d9` |
| `output/local_runs/paper_v_instability_20261010/verify_cs.json` | `c2a74677db9bdd590a5531471dbf95d9833866261cf3859f25de94886c939131` |
| `output/local_runs/paper_v_instability_20261010/verify_minors.json` | `e1ba5b2b9d720275047d6f27273a0f66b7d854ac114a01b684804085e17585e0` |
| `output/local_runs/paper_v_instability_20261010/verify_batched.json` | `1fdcdae9d2f59e00f99511854ab9a6d78154d435361c34e643b273fa1adbfbfc` |
| `output/local_runs/paper_v_instability_20261010/start_states_table.json` | `93954ca838f191a608ea76fcdb35cfabd5828ffaa9aaf852087cd06a4885242f` |
