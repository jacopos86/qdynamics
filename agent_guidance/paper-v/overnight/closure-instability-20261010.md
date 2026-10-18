# The bare closure's instability and its linearized flow

**Revision line (2026-10-18, closure-disposition-scope-20261018-005).**
Text-only wording corrections:

- 13.0 item 7 and 13.3: equal medians of the fired and censored groups are
  reported as a finite observation, not as "does not discriminate".
- 13.5: the two converged keep-only cells are held to a recorded tolerance,
  not exactly.
- The "no attractor" wording of sections 0 and 6 is read with the
  qualification of the consolidation's section 1, item 3.

**Revision note (2026-10-18, dispositions of root reviews 028, 029, the
classical part of 030, and 031).**

- **What changed.** Sections 13 and 14 were amended in place, with no new
  run. The committed versions stay in history: section 13 at `36b686b3`,
  section 14 at `8390a83b`.
- **Where each change is recorded.** Every amendment answers a row of
  `agent_guidance/paper-v/handoffs/DISPOSITION-closure-028-031-claude-20261010.md`.
  Each row names the finding, the decision, the line changed and the
  replacement text.
- **The kinds of change:**
  - the conditional compactness statement is separated from the capped map
    that targets it;
  - "tight" becomes "sharper", and the trace witness is
    `x17 = -omega n / (2 g)`;
  - non-converged counts become batch-step counts;
  - suprema become sampled maxima, records every 400 steps;
  - the selected minors and the retained Gram are named as necessary
    conditions, not physical representability;
  - finite, in-sample and protocol qualifications are attached to the
    ablations, predictors, step-halving ratios, projection comparisons and
    the 3000-unit heating;
  - the accuracy comparisons are kept metric-specific, including the
    `T = 4` Hartree--Fock contradiction in 14.0;
  - the computed-energy target is distinguished from a drive-work
    reference;
  - the hybrid is labelled a capped fallback: 234 infeasible
    re-selections, negative post-event Gram and extension minima, an
    unrecorded cap residual, and a derived, unverified skipped-selection
    cost;
  - the timing protocols are qualified;
  - the execution-provenance and whole-increment limits are stated.
- **What did not change.** Sections 0 to 12 are unchanged; where they
  disagree with 13 or 14, those hold. None of the studies proposed in 13.12
  or 14.7 is selected. The three current strict resume-parity failures and
  root 025 stand.

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

---

# 13. Corrections to sections 8 to 12 after adversarial verification (2026-10-11), and the ablations they asked for

**Branch `claude/paper-v-routes-20260914`; sections 8 to 12 were committed at
`c1c1f9cc`; this section is written at HEAD `840672b4`. Exploratory, not
promoted; no module under `paper_5/src` edited, no default changed, nothing
committed by this round.** Three refuters and a completeness critic checked
the round: R1 section 9, R2 section 10, R3 section 11. Their verdicts are in
the coordinator's verification file (`instability_t2_verify.json`: refuters
0, 1, 2 and the critic), and their reruns are under
`output/local_runs/paper_v_instability_verify_20261010/`
(`R1-escape-statistics/`, `R2-representability-predictor/`,
`R3-confinement/`, `critic/`). Every refuted sentence below is replaced by
the refuter's corrected statement. Where I re-measured, my number is given
beside theirs. On every number I re-measured, theirs reproduce (the only
differences are bootstrap resampling in the third significant figure and
R2's `543` against my `542.8`), so no disagreement is recorded. New receipts
are under `output/local_runs/paper_v_instability_20261010/task2_corrections/`,
and their SHA-256 are in the rehashed `sha256_task2.json` (13.13).

## 13.0 What changes, first

1. **The bare rows of section 9 measure one deterministic trajectory's
   remaining lifetime, not an escape rate of a state.** From the bare
   closure's own states, the time to the `1e4` event is `141.41 - t`
   (exact correlated preparation) or `54.46 - t` (Hartree--Fock
   preparation). The `1e-4` perturbations add `+14.6`, `+3.4`, `+1.2` units
   to the first and change the second by at most `0.18` units.
2. **"Five times lower" compared two different preparations.** The repaired
   chain starts from Hartree--Fock, while the bare row it was set against
   starts from the exact correlated preparation. Against the same preparation
   the factors are `17.1 [16.2, 18.1]` and `33.4 [31.9, 35.0]` at `t = 20`
   and `40`, and both divide by that trajectory's own `54.46 - t`. Against
   the exact driven trajectory from the same Hartree--Fock state the factors
   are `4.6 [4.3, 4.9]`, `1.65 [1.54, 1.78]` and `1.13 [1.07, 1.21]`. For
   the unperturbed states they are `4.1`, `1.3` and `0.94`, so at `t = 60`
   the repaired state is not longer-lived than the exact one.
3. **"Admissible" means three linear equalities, not representable.** The
   perturbations are tangent at first order to the Eq. (22) energy and to Re
   and Im of the shared trace; no positivity is imposed. The bare base states
   already violate the electronic, phonon and joint cones and carry a
   nonzero shared trace (`0.020` to `1.23`), and the perturbation keeps those
   values.
4. **The compactness statement is conditional: it holds for a set, with a
   sharper constant than section 11's, and only with trace control.** It
   bounds every coordinate while a bounded-above Eq. (22) energy, the trace
   pin and every stated inequality actually hold. It says nothing by itself
   about what an executed map achieves. The Gram held positive with the
   energy alone is not compact, because the retained joint Gram subtracts
   `Tr C[q]`. The certificate needs trace control; an exactly zero trace is
   not shown to be the only control that would restore compactness.
   Sentences in 11.3 that said otherwise are withdrawn. No run in sections 8
   to 12 held the energy exactly. This round adds a capped map that targets
   it: it holds the computed Eq. (22) energy at its post-step, pre-map value
   to `1.0e-12`, alternating with the bounds for at most 200 iterations, so
   at some batch steps the bounds or the energy are left unconverged (13.5,
   13.6).
5. **A finite ablation of the capped map (13.5).** With the exact-energy hold,
   dropping any single family of bounds gives no `1e4` event in 600 units
   from either `t = 0` preparation (8 members each). In this finite test, no
   single family is needed to avoid the event. Keeping only the trace pin,
   only the phonon diagonal, both, or the phonon diagonal with the
   anomalous bounds, every member crosses the threshold. Most of those cells
   carry non-converged batch steps, so they describe the capped algorithm,
   not the insufficiency of the exactly enforced intersection; 13.5 gives
   each row's status. With the energy not held, dropping the correlation
   Cauchy--Schwarz bounds lets 7 of 8 exact-correlated members cross the
   threshold (`502` to `576` units) and drives the Hartree--Fock members to
   a sampled maximum absolute coordinate of `2.3e3` by 600. That is a finite
   ablation result, not mathematical necessity.
6. **Cost (13.11).** Measured in one session (minimum of three), the minors
   map costs `1.18` times the bare RK4 step from either preparation
   (`0.165` against `0.140` s per unit, one trajectory, Hartree--Fock). The
   exact energy hold costs `1.5` to `3.1` times. The map is about 21 square
   roots and 22 comparisons per state per step. The cheap repair costs
   `346.8` s per unit in the same session (`t = 0` to `0.4`), or `216.5` on
   the recorded `t = 0` to 60 wall. That is `2.1e3` or `1.3e3` times the
   minors map. The cheap repair's timing includes 8 of 10 failed lift
   re-selections, so the ratios compare the measured protocols, not equally
   accurate methods.
7. **Section 10 (R2, 13.3).** In these 19 ensembles, the timing of the
   retained-Gram loss showed no observed separation of escaping members.
   Under the bare closure the
   retained Gram is negative at the start, or within 12 units of it, in all
   19 ensembles. In the task-11 `t = 20` ensemble, the members with no event
   in 1500 units cross Gram `< -1` at the same median time (`542.8`) as those
   that fire: the fired and censored groups cross at equal medians. That
   finite observation does not establish an absence of predictive
   information. The selected minors and the retained 8-by-8 Gram are necessary
   conditions for physical representability, not a complete test of it.
   This is a finite, in-sample and partly censored observation about that
   necessary condition, not a proof that no starting-state geometry
   predicts escape. Starts satisfying the tested necessary constraints
   escape the bare closure as before (13.10).
8. **The maps (critic, 13.4).** The critic's assessment, carried verbatim,
   is: *the minors clip is a candidate cheap divergence guard, not a
   representability repair, and it does not improve accuracy.* Its "does
   not improve accuracy" compares the clip with the cheap repair. Against
   the bare closure some errors do improve (site-0 RMS over `[0, 4]`:
   `0.0356` against `0.0522` from Hartree--Fock; all-31 RMS over `[0, 20]`:
   `0.3146` against `0.3320`), while the Eq. (22) RMS worsens from the exact
   correlated preparation (13.6). Against the cheap repair, the
   observable-accuracy table (13.6) bears the critic out for every map
   built this round:
   - against the cheap repair, the best map's Eq. (22) RMS is `6.6` to
     `38` times higher at every horizon, and its site-0 RMS is higher at
     `T = 4` and `8`;
   - the one exception is `gram_exact` from Hartree--Fock at `T = 60`, whose
     site-0 RMS is lower (`0.1285` against `0.1604`) but is not
     `h`-converged.
9. **A guard, not a repair (13.8).** Applied only where the maximum absolute
   coordinate passes 3 or the retained Gram falls below `-2.75`, the
   exact-energy minors map prevents every event in 600 units from 32 starts
   that all fire under the bare closure by `566`. It acts at `8.7e-4` and
   `9.7e-3` of member-steps. Its thresholds were tuned on the same
   ensembles. Before its first trigger it is the bare closure, so its
   agreement with bare dynamics is not an accuracy benefit. From
   Hartree--Fock its single trigger, at `t = 37.26`, reached the
   200-iteration cap without bound convergence (13.8). Its zero events are
   finite, censored observations of that capped algorithm, not a
   certificate of exact constraint preservation.

## 13.1 Section 9, corrected

The unperturbed state's own time to the event, my recomputation
(`section9_recheck.py`; DOP853 `rtol 1e-10`, `atol 1e-12`, perturbation size
0, span 1500), matches R1's `rerun_zero` to every printed digit. The
bootstrap uses 20,000 resamples of the 64 members, seed `20261011`; R1's
intervals agree to the third significant figure. "Inverse mean" is
`k / total exposure`, which is exactly `1 / mean` wherever no member is
censored, and only one row is censored. The words "censored
maximum-likelihood" in section 9 are withdrawn: censoring plays no part in
18 of the 19 rows. The after-delay rate is `k / sum(event - earliest)`, with
the sample's earliest event used as a fitted delay, so it inherits the
earliest event's sampling noise. Fresh seeds put a bare exact-correlated
`t = 20` member at `110.5`, below the `113.43` reported as earliest. That
rate is now given for every complete row.

| ensemble (64 members, `1e-4`) | unperturbed time to the event | mean [95% CI] | standard deviation | range | inverse mean | after-delay rate |
|---|---:|---|---:|---:|---:|---:|
| bare Hartree--Fock, `t = 20` | `34.461` (`54.461 - t`) | `34.45 [34.44, 34.46]` | `0.041` | `0.18` | `2.90e-2` | `10.4` |
| bare Hartree--Fock, `t = 40` | `14.461` | `14.46 [14.45, 14.46]` | `0.008` | `0.02` | `6.92e-2` | `65.2` |
| bare exact start, `t = 20` | `121.41` (`141.406 - t`) | `135.99 [128.7, 144.8]` | `33.21` | `182.8` | `7.35e-3` | `4.43e-2` |
| bare exact start, `t = 40` | `101.41` | `104.83 [101.0, 110.0]` | `18.77` | `126.3` | `9.54e-3` | `7.01e-2` |
| bare exact start, `t = 60` | `81.41` | `82.65 [82.1, 83.2]` | `2.33` | `10.4` | `1.21e-2` | `0.399` |
| exact reference, `t = 20` | `127.39` | `128.23 [127.1, 129.5]` | `5.13` | `28.2` | `7.80e-3` | `0.162` |
| exact reference, `t = 40` | `380.25` | `291.69 [275.5, 308.7]` | `68.00` | `297.6` | `3.43e-3` | `8.17e-3` |
| exact reference, `t = 60` | `373.90` | `361.74 [351.0, 373.1]` | `45.78` | `185.0` | `2.76e-3` | `1.46e-2` |
| repaired chain, `t = 20` | `527.08` | `588.69 [558.4, 622.0]` | `133.20` | `712.4` | `1.70e-3` | `5.38e-3` |
| repaired chain, `t = 40` | `481.24` | `482.65 [461.4, 506.2]` | `91.93` | `467.2` | `2.07e-3` | `9.99e-3` |
| repaired chain, `t = 60` | `350.78` | `410.04 [389.2, 433.9]` | `91.57` | `483.0` | `2.44e-3` | `8.78e-3` |
| hyperplane on (task 11), `t = 5` | `394.38` | `413.50 [397.6, 430.8]` | `69.27` | `347.0` | `2.42e-3` | `1.17e-2` |
| hyperplane on, `t = 10` | `199.23` | `199.24 [199.23, 199.25]` | `0.046` | `0.36` | `5.02e-3` | `34.4` |
| hyperplane on, `t = 15` | `1033.96` | `1038.13 [1036.7, 1039.8]` | `6.51` | `42.9` | `9.63e-4` | `9.88e-2` |
| hyperplane on, `t = 20` | no event in 1500 | restricted mean to 1500: `1409.7` (lower bound); Kaplan--Meier median above 1500; 23 of 64 fired, their mean `1248.7` | --- | --- | at most `7.09e-4` | --- |
| hyperplane off (task 13), `t = 5` | `409.25` | `416.36 [411.5, 421.4]` | `20.43` | `89.4` | `2.40e-3` | `3.20e-2` |
| hyperplane off, `t = 10` | `73.78` | `73.80 [73.74, 73.86]` | `0.244` | `1.46` | `1.36e-2` | `2.51` |
| hyperplane off, `t = 15` | `70.75` | `70.16 [69.9, 70.4]` | `1.01` | `4.94` | `1.43e-2` | `0.497` |
| hyperplane off, `t = 20` | `66.21` | `66.22 [66.21, 66.24]` | `0.074` | `0.50` | `1.51e-2` | `5.15` |

Corrections to the text of sections 8 and 9:

- **(a) The bare rows.** "The bare closure's own states ... take `136.0`,
  `104.8`, `82.7`" is replaced. From the bare exact-correlated trajectory's
  state at `t`, the mean time to the event is that trajectory's own
  remaining lifetime `141.41 - t` (`121.4`, `101.4`, `81.4`) plus a delay
  the perturbation adds (`+14.6`, `+3.4`, `+1.2`). From the bare
  Hartree--Fock trajectory's state it is `54.461 - t`, and the perturbations
  change it by at most `0.18` units. Their rise with `t` is `1/(T - t)` by
  construction and says nothing about the states.
- **(b) "Admissible".** Read it as defined in 13.0 item 3; R1's corrected
  statement applies. Measured start states (R1, `start_states_table.json`):
  - bare exact-correlated, `t = 20`: electronic eigenvalue `-0.0320`,
    retained joint Gram `-0.703`, boson-moment eigenvalue `-0.393`,
    `|Tr C| = 0.0203`;
  - bare Hartree--Fock: `|Tr C| = 0.250` at `t = 20`; at `t = 40`, Gram
    `-11.06`, electronic eigenvalue `-1.66`, `|Tr C| = 1.23`;
  - repaired `t = 20`: Gram `-1.65e-4`, boson-moment eigenvalue `-4.71e-4`;
  - task-11 `t = 20`: Gram `-6.5e-5`;
  - exact-reference states: Gram `+0.0038`, `+0.0075`, `+0.0099`.

  The Hartree--Fock preparation itself sits on the cone boundary (`|r| = 1`,
  `N = 0`). Its `1e-4` perturbations therefore start outside `N >= 0` or
  `|r| <= 1` for seeds 1, 3 and 4 (R3).
- **(c) The headline ratio.** "About five times lower than the bare
  closure's own states" is withdrawn. The `4.3`, `4.6`, `5.0`
  (`[4.0, 4.7]`, `[4.3, 4.9]`, `[4.7, 5.3]`) compare the repaired chain
  (Hartree--Fock preparation) with the other preparation's bare trajectory,
  whose event at `141.41` fixes the denominator near `141.41 - t`. The
  same-preparation figures are in 13.0 item 2.
- **(d) Sampling and integrator uncertainty.** Every mean now carries its
  bootstrap interval (table). The fall of the repaired mean with the state's
  time is resolved: differences `106 [67, 146]` and `73 [41, 104]` (R1).
  The unperturbed repaired times fall the same way (`527.1`, `481.2`,
  `350.8`), but the trend rests on three states of one chain. Integrator
  sensitivity (R1, seeds 1 to 8 from the repaired `t = 20` state): at
  `rtol 1e-12`, `atol 1e-14` against `1e-10`, `1e-12`, members move by up to
  `12.3` units and the 8-member mean by `-2.2` (`0.4` percent). Bare
  exact-correlated members move by at most `0.23`.
- **(e) "Not memoryless".** The finding stands. The earliest event lies at
  `0.578` to `1.000` of the mean in every ensemble, against `0.016` expected
  for an exponential sample of 64. Each distribution is the spread of a
  deterministic event time under `1e-4` changes of the start, centred near
  the unperturbed time (R1). The count is restated:
  - five of nineteen ensembles have a standard deviation below one unit
    (`0.008` to `0.244`);
  - in four of them all 64 events fall within one unit (range `0.02` to
    `0.50`);
  - in hyperplane off at `t = 10` the events span `1.46` units.

  "Fixed to within a unit" applies to those four only.
- **(f) The after-delay rate.** Section 9 left it blank for seven rows under
  a "spread below one unit" rule that two of them break: hyperplane on at
  `t = 15` (standard deviation `6.51`) and hyperplane off at `t = 15`
  (`1.01`). The table above gives it for every complete row. It means
  little where the range is below one unit.
- **(g) Hyperplane on at `t = 20`.** 23 of 64 members fire by 1500. The
  restricted mean to 1500 is `1409.7`, a lower bound on the mean, so the
  inverse mean is at most `7.09e-4`. The Kaplan--Meier median exceeds
  1500. The exponential figure `2.55e-4` is withdrawn from the column of
  inverse mean times. Censoring complete ensembles at their own quantiles
  shows that `k / exposure` at 36 percent fired understates a comparable
  inverse mean about 2 to 3 times (R1).
- **(h) What the task-11 and repaired chains share.** R1 notes that
  "hyperplane on" (task 11) and the "repaired chain" carry the same recorded
  repair configuration: 27 keys, Dykstra 120, the Eq. (22) hyperplane at the
  `drive_power` reference. They differ in preparation (exact correlated
  against Hartree--Fock) and in the recorded model-extension backend. The
  section-9 contrast between the repaired chain and "states the repair held
  cones on" is therefore not a contrast between repairs.

## 13.2 Section 11, corrected

- **(a) Two ranking sentences.**
  - "Hands on the shortest-lived" (task 13) is replaced: among the three
    repair chains (repaired, task 11, task 13), the task-13 states are the
    shortest-lived, `66` to `74` units elapsed (absolute `83.8` to `86.2`).
    That is earlier than the bare closure from the same exact-correlated
    preparation (absolute `142.7` to `156.0`). It is not shorter than the
    bare Hartree--Fock states (`34.5` and `14.5` units).
  - "The longest-lived state of all" (task 11) is replaced: the task-11
    states give erratic times, `413.5`, `199.2`, `1038.1` and more than 1500
    for 41 of 64 members at `t = 5`, `10`, `15`, `20`. They do not show
    monotonically that holding the energy and the Gram hands on long-lived
    states.
  - The task-13 trend with energy comes from four states of one chain, in
    which energy and the time of the state are confounded (R3).
- **(b) `cs_offdiag`.** "Lowers the escape rate by 1.0 to 7.6 times" and
  "the escape moves to the phonon blocks" are replaced by R3's measurement:
  - From the Hartree--Fock preparation escape is delayed `5.2` times (all
    fired).
  - From the exact correlated preparation it is delayed at least `3.6` times
    (restricted mean to 600; 15 of 32 fired; Kaplan--Meier median above 600
    against `142`). The `7.6` was an exponential figure on 17 censored
    members and is withdrawn.
  - From the exact `t = 60` state the delay is `1.25` to `1.29` times
    (log-rank `p = 4e-9`).
  - From the repaired `t = 60` state there is no detectable effect (`0.98`
    to `1.01`, `p = 0.87`).
  - The escape is not confined to the phonon blocks. On perturbed members
    `N_qq` is negative from `t <= 14` on (from `t = 0` for seeds 3 and 4).
    Before the event the largest block is the anomalous density, the phonon
    density or the shared trace, and the electronic Bloch norm `|r|` reaches
    `28` to `369`.
  - `cs_offdiag` survivors at 600 are mid-runaway (maximum absolute
    coordinate `257` and `513`). The sentence "surviving members ... 1.41 to
    3.60" applies to the minors maps only.
- **(c) The compactness statement.**
  - Replaced by R3's corrected statement: fixed (or bounded-above) Eq. (22)
    energy together with `N >= 0`, `|r| <= 1`, the twelve 2-by-2 Gram
    minors and `Tr C[q] = 0` gives a compact set. This is a conditional
    statement about the set: it applies to a trajectory only while every one
    of those conditions holds. The lower bound, sharper than section 11's
    (its global optimality is not shown), is
    `E >= -2t + omega |B|^2 - 4 g |B| + omega Tr N - 2 g (sqrt N_00 +
    sqrt N_11)`; section 11's `4 sqrt 2 g` is valid but loose. Every
    coordinate is then bounded:
    - `|B| <= [4g + sqrt(16 g^2 + 4 omega (E_max + 2t + 2 g^2/omega))] / (2 omega)`;
    - `sqrt N_qq <= [2g + sqrt(4 g^2 + 4 omega (E_max + 2t + 5 g^2/omega))] / (2 omega)`;
    - the rest are bounded through these.

    The `1e4` event requires `E >= 4.87e3`. At `E_max = -3.5`:
    `|B| <= 4.90`, `N_qq <= 13.5`, `|A| <= 13.99`, correlations `<= 3.67`.
    My check of the last step: `N_max` with `sqrt(N_max (N_max + 1)) = 1e4`
    gives `E = 4871.8`.
  - **The trace pin is necessary for compactness, and the Gram with the
    energy alone is not compact.** The retained joint Gram subtracts
    `Tr C[q]`. Take `N_00 = N_11 = n`, `x17 = -omega n / (2 g)` (about
    `-0.408 n` at `omega = 0.5`, `g = 0.6124`) and every other coordinate
    zero. The energy `2 omega n + 4 g x17` is then exactly 0, the Gram's
    minimum eigenvalue is `+1`, and `max|x| = n` (R3). This shows that this
    certificate fails without trace control. It does not show that an
    exactly zero trace is the only control that would restore compactness. The 11.3 sentence "The retained joint Gram, which
    implies every one of these bounds, is therefore sufficient with the
    energy but more than is needed" is withdrawn. It now reads: the
    retained joint Gram held positive together with `Tr C[q] = 0` and the
    Eq. (22) energy is sufficient; the Gram with the energy alone is not.
  - **"Each part is measured" is withdrawn.** Sections 8 to 12 ran only
    `none`, `cs_offdiag`, `minors` and `minors_energy`, which shows the full
    set sufficient over 600 units and nothing about which parts are needed.
    13.5 tests that in a finite ablation of the capped map, with each
    row's convergence status.
  - **No run in sections 8 to 12 held the energy exactly.** The
    `minors_energy` restoration through `B` fails whenever the map has
    raised the energy past `B`'s own minimum: the maximum change with
    `hold_energy = True` is `40.99` in `verify_minors.json`. From the exact
    `t = 60` state the drift with restoration (`1.50`) exceeds the drift
    without it (`1.38`). Minors-only heating continues to `+14.5` by 1800
    units without escape (R3).
  - Given the largest recorded energy (sampled every 400 steps, so not a
    bound at every point of the interval), the inequality bounds `max|x|` by
    `14` to `29` at the sampled states of every minors run. So 0 escapes in those runs follows
    from the measured heating and is not an independent finding (R3). The
    informative quantity is the heating rate.
- **(d) "The maps above confine" and "every map above confines without
  the extension"** are restricted to `minors` and `minors_energy`.
  `cs_offdiag` does not confine.
- **(e) Cost.**
  - Replaced: about 21 square roots or complex moduli (22 with the
    restoration) and about 22 comparisons per state per step.
  - The section-11 walls (`0.20` bare against `0.46` minors) came from
    separate runs under different load. R3 timed all four maps in one
    session: minors `1.08` times the bare RK4 step (`0.162` against
    `0.150 s` per unit). My own session is in 13.11.
  - The cheap repair's `216.5 s` per unit is `12,990.087 s` over `t = 0` to
    `60` from a different run (`214.8` over `0` to `20`). Ratios built on it
    run from 470 to 1340 depending on the wall used.
- **(f) Accuracy.**
  - Every "at `t = 20`" in the accuracy table and in 8 to 12 is replaced by
    "over `0 <= t <= 20`". The numbers are the cumulative all-coordinate
    time-RMS over the 31 closed scalar coordinates and every sample of the
    `0.0025` grid from 0 to the horizon (`8,001` samples at 20), against the
    exact driven trajectory from the Hartree--Fock product state: minors
    `0.3145`, minors_energy `0.3136`, bare `0.3320`, cheap repair `0.1537`.
  - From the exact correlated preparation, minors raises the Eq. (22) RMS
    from `0.018` (bare) to `0.160`.
- **(g)** The `clip_ens_*.log` files, `clip_accuracy_*.log`, `escape2.log`
  and every file of this round are now in `sha256_task2.json` (13.13).
- **(h)** The "128 member-runs" are 64 distinct initial conditions (seeds 1
  to 16 at each of four starts), each run under `minors` and under
  `minors_energy`.

## 13.3 Section 10, corrected (refuter R2)

**The quantity, named.** "Gram" in section 10 is the minimum eigenvalue of
the 8-by-8 affine Gram of `(1, b0, b1, b0^dag, b1^dag, sigma_x, sigma_y,
sigma_z)`, with the correlation shared trace removed (`run_unrepaired.py`).
It is not a 16-feature matrix and not the 36-row extension. Positivity of
this Gram and of the selected minors are necessary conditions for a
physical moment state. They are not a complete test of physical
representability, and "representability" in this section means only the
tested necessary condition. R2 reproduced
every tabled cell of section 10 from the stored first crossings, and its
16-member reruns matched bitwise. Every refuted sentence is replaced:

- **"Loses joint representability long before its event"** is replaced.
  - Under the bare closure the retained Gram is negative at the start state,
    or within `0.5` to `12` units of it, in every one of the 19 ensembles.
    The repaired chain's own states are already at `-1.65e-4`, `-1.90e-4`
    and `-2.65e-6`. The exact-reference states start at `+3.8e-3`,
    `+7.5e-3`, `+9.9e-3` and cross 0 within `1.5`, `0.75` and `1.2` units.
  - "Below `-1`" is a depth threshold, not loss of representability. It is
    crossed a median `175` to `318` units before the event in the five
    tabled ensembles (member minimum `50`), and only `14.5` to `51` units
    before in the bare Hartree--Fock, hyperplane-on `t = 10` and
    hyperplane-off `t >= 10` ensembles. Across all 19 ensembles the median
    lead is `14.5` to `724.2`.
  - The Hartree--Fock preparation has a retained Gram of rank 4 of 8 at
    `t = 0`, and its bare trajectory is below `-1e-8` by `t = 0.015`.
- **"Carries no information about when the event comes"** is replaced.
  - Within one ensemble (64 perturbations of size `1e-4` around one state)
    the first time the Gram falls below `-1` does not order the members'
    events: rho `-0.04` to `0.25`, `p` `0.047` to `0.95`. The section-10
    table and the receipts give `-0.0443` for the repaired `t = 40`
    ensemble; this line printed `-0.01` before the 2026-10-18 revision.
  - In two of the five tabled ensembles the members cross within one to
    four `0.05`-unit samples. There the correlation is undefined in
    substance, and "lead spread equals event spread" is an identity.
  - Across the 19 starting states, the median time to Gram `< -1` is
    rank-correlated `0.905` with the median event time (`p = 1.0e-7`; my
    recomputation from the stored members gives the same `0.905`). This is
    partly by construction, since every crossing precedes its event.
- **"No earlier, and no better, than the maximum absolute coordinate passing
  4"** is replaced.
  - At section 10's own thresholds, Gram `< -5` leads the coordinate
    passing `4` in five of five ensembles, by `1.7` to `6.5` units.
  - With thresholds tuned on the same quantity, the earliest crossing that
    orders the members' events (rank correlation at least `0.9` over 64
    members) comes a median `18` to `38` units before the event for the
    retained Gram (thresholds `-2.75` to `-4.5`). For the maximum absolute
    coordinate it comes `12` to `26` units before (thresholds `2.4` to `4`).
  - At equal predictive skill (`0.8`, skill being one minus the mean
    absolute error of the lead about its median over that of the event
    about its median) the Gram is `1.2` to `6.2` units earlier (receipt
    `d_threshold_scan.json`, low end `1.228`; this line printed `1.3`
    before the 2026-10-18 revision).
  - Neither leads by more than about 40 units, under 10 percent of the 281
    to 577 units to the event. These leads are for the five ensembles of
    section 10's table (of 19), on which the thresholds were tuned, so they
    are in-sample. Section 10's fixed-threshold table covers 9 of the 23
    stored thresholds.
- **"Unrelated"** (the early crossings) is replaced. They order the events
  weakly or not at all (rho `-0.06` to `0.42` within an ensemble). The
  positive values in the repaired `t = 60` and exact `t = 60` ensembles
  (`0.42` and `0.31`) are distinguishable from zero (`p = 5e-4` and
  `0.013`).
- **"The loss predicts that the bare closure will escape"** is withdrawn.
  Every member measured loses joint representability under the bare closure
  within 12 units of its start, so no contrast case exists. In the task-11
  `t = 20` ensemble, 38 of the 41 members without an event in 1500 units
  cross Gram `< -1`, at a median of `542.8` units; the 23 members that fire
  cross at a median of `542.8`, the same time (my recomputation; R2 reports
  `543`). In these ensembles, loss of the retained-Gram condition is a
  property of the bare flow from every tested state. In the task-11
  ensemble the fired and censored groups cross at equal medians. That is a
  finite, in-sample observation with
  censoring (23 of 64 fired by 1500). It is not a proof that no measure of
  representability predicts escape from any state.
- **"The bare `t = 20` value `-1.63` at 200 before"** rests on 4 of 64
  members. The 64-member value at 100 before is `-0.99`, and that ensemble
  starts at Gram `-0.70`.
- **The `1.1e-16` agreement** of the Cauchy--Schwarz margin with the module
  holds on bounded states. On the bare Hartree--Fock trajectory it is
  `6e-16` relative (`3e-8` absolute at margins of `6.5e7`).
- **Beside the chain result.** The repaired chain holds the off-diagonal
  Cauchy--Schwarz minors over `t = 0` to `60`, but not the full Gram: the
  retained Gram is below 0 at `87` percent of 241 samples (every `0.25`),
  below `-1e-4` at `29` percent, minimum `-7.96e-4` at `t = 51.75` (R2).

## 13.4 Section 11's maps, corrected (critic and R3)

- **What the minors map targets after every step.** These are held to the
  recorded residual: at most `3.6e-12` without the energy hold, and with it
  `1e-12` except at the non-converged batch steps listed in 13.5. The
  `maps2` and `run_spec.py` runs (13.6 to 13.9) record the energy residual,
  and `obs_accuracy` the largest alternation count, but no bound residual.
  The families are:
  - the Bloch ball, so the electronic `2 x 2` density is positive
    semidefinite;
  - the `2 x 2` phonon density `N`, positive semidefinite (`N_qq >= 0` and
    `|N_01|^2 <= N_00 N_11`);
  - `|A_qq|^2 <= N_qq (N_qq + 1)` and the `|A_01|` bound;
  - six correlation Cauchy--Schwarz bounds along the fixed Pauli `x`, `y`,
    `z` axes;
  - `Tr C[q] = 0`.

  **What it does not hold:**
  - the retained 8-by-8 joint Gram: below `-1e-4` at `64` to `75` percent of
    unit records over `t <= 60` from Hartree--Fock, minimum `-0.366`; at
    `90` to `98` percent from the exact correlated preparation, minimum
    `-0.63` (`obs_accuracy_*.json`; the critic's `t <= 20` figures are
    `77`/`97` percent and `-0.37`/`-0.77`);
  - the full normal/anomalous boson moment matrix `[[N^T, A^*], [A, 1 + N]]`
    (minimum `-0.142` from Hartree--Fock, `-0.302` from exact correlated);
  - the Eq. (22) energy.

  "Every second-order bound" and "positivity bounds on every block" in
  section 11 are withdrawn and replaced by this list.
- **The tested always-on maps behave as first order in `h`.** This is an
  observation on one unperturbed trajectory per preparation at three step
  sizes. It is not an order theorem, a continuum limit, or a statement about
  the accuracy of the underlying closure. With a map after every RK4 step,
  the state difference between `h` and `h/2`, divided by the difference
  between `h/2` and `h/4`, is `1.91` to `2.39` at `t = 20` for every map,
  against `16.0` for the bare closure (`order_check.json`; Euclidean
  differences, one unperturbed trajectory per preparation). At `t = 20` the
  `h` to `h/2` difference is `1.1e-2` (minors, Hartree--Fock) and `1.2e-3`
  (exact correlated), against `2.3e-9` and `4.5e-10` bare. The critic's
  `8.3e-3` and `6.5e-4` are the same quantity in its own norm. The scores
  move by at most `2.4e-3` between `h` and `h/2` at `T <= 20` and by at most
  `2e-3` at `T = 60` (table in 13.6), so the accuracy below is
  `h`-converged to that level. There are two exceptions. `gram_exact` is
  not `h`-converged: its state moves by `0.38` to `0.47` between `h` and
  `h/2` and its scores by up to `0.041` (13.7). The `minors_energy` map from
  the exact correlated preparation moves by `0.39` at `t = 60`, with scores
  moving by up to `0.031` at `T = 60`.
- **The map acts at every step.** The fraction of member-steps clipped is
  `1.0` in all eight minors ensembles, so the minors flow differs from the
  bare closure everywhere.
- **Pulse inputs.** The sentence at section 11 had the pulse inputs
  swapped: `0.24` from Hartree--Fock and `0.22` from the exact correlated
  preparation.
- **`cs_offdiag` from exact `t = 60`.** It shifts the events later (median
  `366` to `469`, log-rank `p = 4e-9`), while the exponential rate ratio
  `1.29` is not a valid summary. The non-parametric statement replaces it.
- **Task 13, "faster as it heats".** From `t >= 10` its states escape at
  absolute `t = 83.8` to `86.2`, nearly fixed, and earlier than the bare
  closure from the same preparation (`142.7` to `156.0`).
- **The critic's assessment, carried verbatim:** *the minors clip is a
  candidate cheap divergence guard, not a representability repair, and it
  does not improve accuracy.* The critic compared the clip with the cheap
  repair. Against the bare closure the clip improves some errors and
  worsens others; 13.6 keeps those comparisons separate.

## 13.5 A finite ablation of the bound families (the necessity ablation and run 2)

**Protocol.** 8 members per cell: seeds 1 to 8 of the admissible direction
(`constrained_initial_direction`, size `1e-4`, as in section 11) around the
`t = 0` state of each preparation. RK4 at `h = 0.0025`, the map after every
step, 600 units, event at `max|x_i| = 1e4` (`minors_family.py`,
`run_family.py`). The families are: `trace` (`Tr C[q] = 0`), `electronic`
(the Bloch ball), `n_diag` (`N_qq >= 0`), `n01` (`|N_01|^2 <= N_00 N_11`),
`a` (the anomalous bounds) and `corr` (the six correlation
Cauchy--Schwarz bounds). "Energy not held" applies the remaining families
once. "Exact energy" alternates them with the exact root of the Eq. (22)
energy along its gradient until both the bound violation and the energy
residual are at most `1e-12` (at most 200 alternations; 13.6). The
exact-correlated drop-`trace` cell with exact energy ran through `maps2`'s
identical spec in 100-unit batches (`gate_drop_trace.json`: difference `0.0`
over 5 units on both preparations). That driver records the energy
residual but not the count of non-converged steps.

Each cell gives the members fired by 600 and the maximum absolute
coordinate at 600 over survivors. In brackets is its largest value at the
records taken every 400 steps, one per unit: a sampled maximum, not a bound
over the whole interval. For energy
not held, the largest change of the Eq. (22) energy from `t = 10` (after the
pulse) to 600.

| family dropped | Hartree--Fock, energy not held | Hartree--Fock, exact energy | exact correlated, energy not held | exact correlated, exact energy |
|---|---|---|---|---|
| none (full set) | 0; `1.91` (`2.21`); `+1.05` | 0; `1.99` (`2.22`) | 0; `3.60` (`4.46`); `+5.57` | 0; `1.57` (`1.81`) |
| `trace` | 0; `5.76` (`10.2`); `+5.53` | 0; `2.00` (`2.64`) | 0; `14.7` (`14.7`); `+11.1` | 0; `2.20` (`2.38`) |
| `electronic` | 0; `1.66` (`2.21`); `+0.98` | 0; `2.00` (`2.22`) | 0; `1.75` (`1.85`); `+0.040` | 0; `1.44` (`1.85`) |
| `n_diag` | 0; `1.73` (`2.32`); `+1.10` | 0; `1.80` (`2.21`) | 0; `3.85` (`8.73`); `+9.37` | 0; `1.57` (`1.81`) |
| `n01` | 0; `1.64` (`2.21`); `+0.95` | 0; `2.00` (`2.21`) | 0; `1.68` (`2.65`); `+2.43` | 0; `1.71` (`1.84`) |
| `a` | 0; `27.3` (`39.4`); `+7.80` | 0; `3.23` (`3.80`) | 0; `172` (`196`); `+190` | 0; `1.62` (`1.98`) |
| `corr` | 0; `2.33e3` (`2.59e3`); `+1.34e3` | 0; `2.22` (`2.29`) | 7 fired at `502` to `576`; survivor `7.4e3` (`9.96e3`); `+5.0e3` | 0; `1.64` (`1.81`) |

With exact energy the Eq. (22) energy changes by at most `4.6e-10` from
`t = 10` to 600 in every drop-one cell. Three Hartree--Fock cells did not
converge at every step: drop `a` at 120 batch steps (largest remaining
bound violation `6.1e-3`), drop `n_diag` at 10 (`5.8e-4`) and drop `trace`
at 7 (`5.3e-8`). These are batch-step counts. `minors_family.py` counts one
non-converged step whenever the whole batch fails to converge, and the
number of failing members was not retained. Every other drop-one cell with
exact energy held the kept bounds to `1e-12` and the energy to `1e-12` at
every step. The same map from the exact `t = 60` state and the repaired
`t = 60` state (all families, exact energy, 8 members, 600 units) gives 0
fired, sampled maxima `2.42` and `1.97`. The exact-state run did not
converge at 305 batch steps (violation `1.2e-7`). The exact-correlated
drop-`trace` cell ran through `maps2`, which records neither bound
residuals nor non-converged steps, so its enforcement is established only
for the energy residual.

Keep-only cells, exact energy (the families not named are dropped):

| families kept | Hartree--Fock | exact correlated |
|---|---|---|
| `trace` | 8 fired, `50.3` to `50.6` | 8 fired, `159` to `172` |
| `n_diag` | 8 fired, `76.9` to `85.1` | 8 fired, `156` to `291` |
| `trace`, `n_diag` | 8 fired, `48.2` | 8 fired, `159` to `171` |
| `n_diag`, `a` | 8 fired, `165` to `205` | 8 fired, `164` to `221` |
| `trace`, `n_diag`, `a` | 0 fired; `1.97` (`2.29`) | 8 fired, `231` to `507` |
| `corr` | 0 fired; `169` at 600, still growing | 0 fired; `2.07` (`2.28`) |
| `corr`, `electronic` | 0 fired; `26.6` (`39.4`) | 0 fired; `2.14` (`2.33`) |

Each keep-only cell's convergence, as non-converged batch steps (the
number of failing members was not retained) / largest recorded bound
violation / largest energy residual after the map (`family_*.json`):

| families kept | Hartree--Fock | exact correlated |
|---|---|---|
| `trace` | 1964 / `0` / `1.5e-11` | 0 / `0` / `9.1e-13` |
| `n_diag` | 9 / `1.0e-12` / `1.4e-12` | 846 / `1.0e-12` / `1.1e-11` |
| `trace`, `n_diag` | 0 / `1.0e-12` / `9.1e-13` | 2 / `1.0e-12` / `1.4e-12` |
| `n_diag`, `a` | 3487 / `1.0e-12` / `2.2e-11` | 4583 / `1.0e-12` / `2.2e-11` |
| `trace`, `n_diag`, `a` | 0 / `2.2e-16` / `4.0e-15` | 26 / `1.0e-12` / `3.6e-12` |
| `corr` | 2 / `2.1e-10` / `1.0e-12` | 0 / `1.0e-12` / `9.9e-13` |
| `corr`, `electronic` | 2 / `2.1e-10` / `1.0e-12` | 1 / `3.5e-12` / `1.0e-12` |

Seven of the nine fired keep-only cells carry non-converged batch steps. In
the fired cells the residual left at those steps is in the energy (up to
`2.2e-11`), while the kept bounds are held to `1e-12`. These rows are a
finite test of the capped algorithm at those residuals. A capped map that
leaves kept constraints unconverged cannot establish that the exactly
enforced intersection is insufficient. Two fired cells converged at every
step: Hartree--Fock keep `trace`, `n_diag`, and exact correlated keep
`trace`.

What this measures:

- With the exact-energy hold, in this finite test (8 members per cell, 600
  units, `1e4` threshold), dropping any single family gives 0 events and a
  sampled maximum of at most `3.80`. Three Hartree--Fock cells and the exact
  `t = 60` cell carry non-converged batch steps (above). The
  exact-correlated drop-`trace` cell has no bound-residual receipt.
- With the capped alternation, keeping only the trace pin, only `N_qq >= 0`,
  both, or `N_qq >= 0` with the anomalous bounds lets every member cross the
  threshold. The three together give no event from Hartree--Fock and 8
  events from exact correlated. Two cells converged at every step: the
  Hartree--Fock `trace`, `n_diag` cell and the exact-correlated `trace`
  cell. In those two the capped algorithm held the kept bounds and the
  energy to its recorded tolerance (`1e-12`) at every step, for the 8
  members and the 600-unit window, and the event still occurred.
  Finite-tolerance enforcement is not exact set membership, so this does
  not establish insufficiency of the exactly enforced set. The conditional
  compactness algebra of 13.2 (c) is unaffected.
- The correlation bounds alone, with the exact-energy hold, give no event in
  600. They do not keep the coordinates small from Hartree--Fock (`169` at
  600).
- With the energy not held (one application of the remaining families per
  step, residual at most `3.6e-12`), dropping the correlation bounds lets 7
  of 8 exact-correlated members cross the threshold and drives the
  Hartree--Fock members to a sampled `2.3e3`. That is a finite ablation
  result, not mathematical necessity. Dropping the anomalous bounds drives the heating to
  `+190` (exact correlated) and `+7.8` (Hartree--Fock) without an event in
  600. Dropping the trace pin raises it to `+11.1` and `+5.5`.
- From the exact correlated preparation the Bloch-ball clip is the source of
  the minors map's heating: without it the energy moves `+0.040` from
  `t = 10` to 600, against `+5.57` with it.

## 13.6 The exact energy hold, and the observable accuracy (run 1)

**The hold.** `minors_exact` applies the minors families, then the exact
root of `E(x + s grad E) = E_target` (a quadratic in `s`, since Eq. (22) is
quadratic), alternating until the bound violation and `|E - E_target|` are
both at most `1e-12`, for at most 200 alternations. `E_target` is the
computed energy after the RK4 step, before the map; it is not taken from the
exact reference. The map itself does no work. By changing the state, though,
it can change the work the pulse does afterwards. Over every run of this
round the energy residual after the map is at most `1.0e-12`, apart from
the keep-only cells listed in 13.5 (up to `2.2e-11`). The bounds are not
always converged (13.5).

**Protocol.** One unperturbed trajectory per preparation, RK4 at
`h = 0.0025` and at `h/2`, the map after every step, to `t = 60`
(`obs_accuracy.py`). Scored on the `0.0025` grid against the exact driven
trajectory from the same preparation:

- Hartree--Fock: the exact trajectory stored beside the repaired chain;
- exact correlated: the same exact propagation
  (`run_divergence_segment._reference_coordinates`, DOP853 `rtol 1e-9`,
  `atol 1e-11`, phonon cutoff 16), which matches the task-13 segments'
  stored exact coordinates to `3.6e-15` over `t <= 20`.

Each cell is the cumulative time-RMS over every grid sample in `[0, T]` of
the site-0 occupation `rho[0,0]`, of the Eq. (22) energy, and over all 31
closed scalar coordinates (`_accuracy_metrics`). The cheap-repair rows are
the stored chains rescored with the same function
(`cheap_repair_accuracy.py`): the repaired chain from Hartree--Fock to
`t = 60`, and the task-11 chain (same configuration) from the exact
correlated preparation, which was run only to `t = 20`. The maps are
`none` (the bare closure), `minors`, `minors_energy` (section 11's
restoration through `B`), `minors_exact`, `gram` and `gram_exact` (13.7),
`boson_minors_exact` (13.7) and `triggered_minors_exact` (13.8). The last
column is the largest change of any of the twelve scores between `h` and
`h/2` (`obs_accuracy_table.md`).

**Hartree--Fock preparation**

| variant | T = 4 | T = 8 | T = 20 | T = 60 | event (elapsed) | largest score change, h to h/2 |
|---|---|---|---|---|---:|---:|
| `none` | 0.0522 / 0.0421 / 0.1296 | 0.0823 / 0.0440 / 0.2937 | 0.1682 / 0.0452 / 0.3320 | event | 54.46 | 2.1e-11 |
| `minors` | 0.0356 / 0.0436 / 0.1313 | 0.0659 / 0.0466 / 0.2969 | 0.1654 / 0.0480 / 0.3145 | 0.2715 / 0.0858 / 0.3576 | none to 60 | 1.9e-03 |
| `minors_energy` | 0.0358 / 0.0425 / 0.1310 | 0.0660 / 0.0456 / 0.2960 | 0.1654 / 0.0473 / 0.3136 | 0.2437 / 0.0353 / 0.3458 | none to 60 | 4.1e-04 |
| `minors_exact` | 0.0356 / 0.0437 / 0.1313 | 0.0659 / 0.0469 / 0.2970 | 0.1655 / 0.0487 / 0.3146 | 0.2890 / 0.0495 / 0.3668 | none to 60 | 4.2e-04 |
| `gram` | 0.0359 / 0.2292 / 0.1158 | 0.0470 / 0.3274 / 0.2670 | 0.0859 / 0.3841 / 0.2584 | 0.1979 / 0.4902 / 0.3048 | none to 60 | 5.3e-04 |
| `gram_exact` | 0.0464 / 0.0358 / 0.1085 | 0.0673 / 0.0461 / 0.2323 | 0.0839 / 0.0512 / 0.4595 | 0.1285 / 0.0534 / 0.5479 | none to 60 | 1.3e-02 |
| `boson_minors_exact` | 0.0448 / 0.0409 / 0.1057 | 0.0621 / 0.0428 / 0.2478 | 0.1611 / 0.0439 / 0.2651 | 0.2609 / 0.0443 / 0.3042 | none to 60 | 9.4e-05 |
| `triggered_minors_exact` | 0.0522 / 0.0421 / 0.1296 | 0.0823 / 0.0440 / 0.2937 | 0.1682 / 0.0452 / 0.3320 | 0.3155 / 0.0457 / 0.4294 | none to 60 | 4.6e-11 |
| cheap repair (repaired chain, 27 keys, Dykstra 120, hyperplane on) | 0.0070 / 0.0011 / 0.0108 | 0.0331 / 0.0011 / 0.0372 | 0.0736 / 0.0012 / 0.1537 | 0.1604 / 0.0012 / 0.2523 | none | --- |

**Exact correlated preparation**

| variant | T = 4 | T = 8 | T = 20 | T = 60 | event (elapsed) | largest score change, h to h/2 |
|---|---|---|---|---|---:|---:|
| `none` | 0.0296 / 0.0276 / 0.1109 | 0.0448 / 0.0222 / 0.1382 | 0.1431 / 0.0182 / 0.1932 | 0.3054 / 0.0162 / 0.3248 | none to 60 | 2.8e-10 |
| `minors` | 0.0269 / 0.1008 / 0.1081 | 0.0506 / 0.1096 / 0.1214 | 0.1586 / 0.1603 / 0.1686 | 0.2810 / 0.3070 / 0.2559 | none to 60 | 1.3e-03 |
| `minors_energy` | 0.0362 / 0.0793 / 0.1299 | 0.0459 / 0.0877 / 0.1336 | 0.1394 / 0.0924 / 0.1578 | 0.2426 / 0.0992 / 0.2122 | none to 60 | 3.1e-02 |
| `minors_exact` | 0.0192 / 0.0352 / 0.1059 | 0.0465 / 0.0353 / 0.1148 | 0.1347 / 0.0353 / 0.1537 | 0.3004 / 0.0354 / 0.2513 | none to 60 | 1.6e-04 |
| `gram` | 0.0218 / 0.4429 / 0.1225 | 0.0521 / 0.6315 / 0.1326 | 0.2036 / 0.9321 / 0.2229 | 0.2105 / 1.0847 / 0.2231 | none to 60 | 1.2e-03 |
| `gram_exact` | 0.0991 / 0.1367 / 0.4508 | 0.1525 / 0.1313 / 0.4890 | 0.2222 / 0.1280 / 0.5257 | 0.2153 / 0.1265 / 0.5279 | none to 60 | 4.1e-02 |
| `boson_minors_exact` | 0.0192 / 0.0352 / 0.1059 | 0.0497 / 0.0353 / 0.1155 | 0.1856 / 0.0353 / 0.1931 | 0.3024 / 0.0354 / 0.2702 | none to 60 | 4.8e-04 |
| cheap repair (task 11, same configuration) | 0.0116 / 0.0040 / 0.0093 | 0.0281 / 0.0049 / 0.0330 | 0.0681 / 0.0053 / 0.0560 | not run | none | --- |

- No map comes within `6.6` times the cheap repair's Eq. (22) RMS at any
  horizon. The ratio of the best map to the cheap repair is `29` to `38`
  from Hartree--Fock and `6.6` to `8.7` from the exact correlated
  preparation.
- On the site-0 occupation the cheap repair is better at `T = 4` and `8`
  from both preparations (best map `5.1` and `1.4` times higher from
  Hartree--Fock, `1.7` and `1.6` from exact correlated) and at `T = 20` from
  exact correlated (`2.0` times). From Hartree--Fock at `T = 20` the
  Gram-cone maps come within `1.1` times (`0.0839` against `0.0736`), and
  at `T = 60` `gram_exact` is lower (`0.1285` against `0.1604`). Its score
  is not `h`-converged (it moves by `0.013` between `h` and `h/2`).
- Against the bare closure, the minors map lowers the site-0 RMS at `T = 4`
  (`0.0356` against `0.0522` from Hartree--Fock, `0.0269` against `0.0296`
  from exact correlated) and leaves it within `0.003` at `T = 20` from
  Hartree--Fock. From the exact correlated preparation it raises the Eq. (22)
  RMS from `0.018` to `0.160` at `T = 20`. The exact hold brings that to
  `0.035`.
- The Eq. (22) RMS that remains under the exact hold (`0.035` from exact
  correlated, `0.044` to `0.050` from Hartree--Fock) is consistent with a
  difference between the work the pulse does on the closure and on the
  exact state. At `t = 1` the closure's energy is `-3.567` with or without
  the map, against the exact `-3.597`. That rounded agreement at one time
  does not establish equal work histories.
  - An energy-neutral map changes the state, and so can change the work the
    drive does later.
  - The hold targets the computed post-step energy, not an independently
    accumulated drive-work reference.
  - So the attribution of the whole RMS to pulse work is not proved.

## 13.7 The Gram-cone map and the boson moment map (runs 3 and 4)

**Definitions** (`maps2.py`):

- `gram`: set the shared trace to 0 (`x17 = x18 = 0`), form the retained
  8-by-8 joint Gram, take one eigendecomposition, clip the eigenvalues at 0,
  and map back by least squares, averaging the Gram entries that share a
  coordinate. One pass per step, not iterated to the cone.
- `gram_exact`: `gram`, then one exact energy root (shared trace pinned).
  The two are not alternated, so the energy step can leave the cone.
- `boson_minors_exact`: one projection of the 4-by-4 boson moment matrix
  `[[N^T, A^*], [A, 1 + N]]` onto the positive semidefinite cone (same
  clip-and-map-back), then `minors_exact`. The boson matrix is not
  re-imposed inside the alternation.

**Confinement**, 8 members (seeds 1 to 8, `1e-4`) from each `t = 0`
preparation, 600 units:

| map | preparation | fired | maximum absolute coordinate at 600 (in brackets, the largest sampled value; records every 400 steps) | Eq. (22) change, `t = 10` to 600 | smallest sampled retained Gram eigenvalue | smallest sampled boson-matrix eigenvalue |
|---|---|---|---|---|---|---|
| `gram` | Hartree--Fock | 0 | `1.54` (`2.01`) | `+0.77` | `-6.2e-4` | `-6.4e-5` |
| `gram` | exact correlated | 0 | `1.82` (`2.14`) | `+1.81` | `-1.4e-3` | `+2.1e-5` |
| `gram_exact` | Hartree--Fock | 0 | `1.84` (`2.22`) | `4.9e-11` | `-1.76` | `-2.7e-3` |
| `gram_exact` | exact correlated | 0 | `1.88` (`1.90`) | `6.1e-11` | `-2.20` | `-5.6e-3` |
| `boson_minors_exact` | Hartree--Fock | 0 | `1.65` (`2.33`) | `2.8e-12` | `-0.60` | `-1.8e-3` |
| `boson_minors_exact` | exact correlated | 0 | `1.84` (`2.45`) | `2.3e-9` | `-0.73` | `-1.4e-3` |

- `gram` holds the retained Gram to `-6.2e-4` (Hartree--Fock) and
  `-1.4e-3` (exact correlated), below `-1e-4` at `1.3` and `2.3` percent of
  unit records. It heats (`+0.77`, `+1.81`) and has no event in 600 units
  (8 members per preparation). It is one clip-and-map-back pass per step,
  which need not land in the cone, and the retained Gram stays negative.
- `gram_exact` holds the energy and has no event in 600 units. The energy
  root after the one-pass clip destroys positivity again: the Gram falls to
  `-1.76`
  and `-2.20` and is below `-1e-4` at `99.7` and `99.8` percent of unit
  records. It is not `h`-converged: its state moves by `0.05` to `0.47`
  between `h` and `h/2` at `t = 4`, 20 and 60 (largest coordinate
  difference), and its scores by up to `0.041`.
  It is a map onto the energy surface that passes through the Gram cone, not
  a map onto the intersection.
- `boson_minors_exact` holds the boson matrix to `-1.8e-3` (Hartree--Fock)
  and `-1.4e-3` (exact correlated, sampled minima) and has no event in 600
  units from either preparation. The retained
  Gram is not held (`-0.60`, `-0.73`). The exact-correlated cell ran in
  150-unit batches (13.13).
- Order in `h` (`order_check.json`): every tested map, `gram` and
  `boson_minors_exact` included, behaves as first order on the tested
  trajectories (ratios `1.91` to `2.39`). No order theorem is shown.
- None of these names means a projection onto a fully enforced
  intersection, or a physical repair. `run_spec.py` records no bound
  residual or alternation count for these cells.

## 13.8 A triggered guard (run 5)

**Definition.** `triggered_minors_exact` integrates the bare closure and
applies `minors_exact` only to members whose maximum absolute coordinate
exceeds 3 or whose retained Gram minimum is below `-2.75`. These are the
earliest-ordering thresholds of R2's scan (13.3), chosen on the same
ensembles, so the guard is tuned on the states it is tested on.

| start | members | fired by 600 | fraction of member-steps mapped | maximum absolute coordinate at 600 (in brackets, the largest sampled value; records every 400 steps) | Eq. (22) change over 600 |
|---|---:|---:|---:|---|---:|
| repaired chain, `t = 60` | 16 | 0 | `8.7e-4` | `2.35` (`3.28`) | `1.5e-10` |
| exact reference, `t = 60` | 16 | 0 | `9.7e-3` | `3.06` (`3.77`) | `3.9e-9` |

The same 16 starts (seeds 1 to 16, `1e-4`) all fire under the bare closure
(DOP853, section 9): at `307` to `566` units (repaired) and `312` to `478`
(exact).

From the Hartree--Fock preparation (one unperturbed trajectory, 13.6) the
guard acts at a single step, at `t = 37.26`, where the bare trajectory's
retained Gram first falls below `-2.75`. The maximum absolute coordinate
stays below 3 throughout, at most `2.56`. At that step the alternation
reaches its 200-iteration cap, with the energy residual 0 and the bounds
not converged to `1e-12`. That one map moves the state off the bare
trajectory, which fires at `54.46`, and no event follows to `t = 60`. Its
scores equal the bare closure's to `T = 20` (13.6). At `T = 60` its
all-coordinate RMS is `0.429`, against `0.346` to `0.367` for the always-on
minors maps. Its scores move by `4.6e-11` between `h` and `h/2`.

Before its first trigger the triggered guard is the bare closure, so its
agreement with the bare scores to `T = 20` is not an independent accuracy
benefit. Its zero events over 600 units are finite, censored observations
of a capped algorithm whose single Hartree--Fock trigger did not converge.
They do not certify exact constraint preservation. `run_spec.py` records no
alternation count for the two 16-member ensembles.

## 13.9 Heating with the minors map alone, to 3000 units (run 6)

`minors` (energy not held), 8 members (`1e-4`), 3000 units, in 600-unit
batches. Eq. (22) change from `t0 + 10`, median / largest over members:

| start | 600 | 1200 | 1800 | 2400 | 3000 | largest sampled maximum absolute coordinate (records every 400 steps) |
|---|---|---|---|---|---|---:|
| bare exact correlated, `t = 0` | `+2.33` / `+5.57` | `+3.46` / `+7.33` | `+6.23` / `+53.6` | `+21.4` / `+323` | `+66.4` / `+319` | `285` |
| exact reference, `t = 60` | `+0.34` / `+1.38` | `+1.74` / `+3.57` | `+2.54` / `+57.6` | `+3.34` / `+76.5` | `+3.89` / `+95.4` | `124` |

No member fires in 3000 units (8 members per start, `1e4` threshold). The
recorded heating has not levelled off by 3000. That does not show heating
without bound. In both
ensembles some members run away in energy between 1200 and 1800 units (largest
change `+339` exact correlated, `+115` exact `t = 60`, at the records) and
reach sampled maximum absolute coordinates of `285` and `124`. By the bound
of 13.2 (c), the `1e4` event needs `E >= 4.87e3`. The energy is sampled, so
the records do not bound it at every point, and the bound applies only at
the sampled states. Over 3000 units it does not escape,
but it does not confine to the `O(1)` set either: at 3000 the
median maximum absolute coordinate is `73` (exact correlated) and `2.4`
(exact `t = 60`).

## 13.10 Escape from starts satisfying the tested necessary constraints (run 7)

From the repaired chain's and the exact reference's states at `t = 20`:
`x0 + eps u` (seeds 1 to 16, `eps = 1e-4` and `1e-3`), carried into a set
satisfying a tested necessary condition (not a certificate of physical
representability), then the bare closure (DOP853 `rtol 1e-10`,
`atol 1e-12`, event `1e4`, span 1500; `representable_escape.py`). The two
sets are:

- **minors**: every family of 13.5 applied once. The repaired state's
  retained Gram stays at `-4.9e-5`, because the minors do not imply the
  Gram.
- **Gram cone**: `gram_map` repeated until the minimum eigenvalue is at
  least `-1e-12` (77 repetitions for the unperturbed repaired state). This
  moves the Eq. (22) energy by up to `5.0e-4`.

| start | set | `eps` | fired | mean elapsed (sd) | projected unperturbed start | displacement by the projection |
|---|---|---:|---:|---|---:|---:|
| repaired `t = 20` | minors | `1e-4` | 16 | `420.0` (`73.3`) | `414.8` | `5.0e-4` |
| repaired `t = 20` | minors | `1e-3` | 16 | `501.5` (`103.6`) | `414.8` | `7.9e-4` |
| repaired `t = 20` | Gram cone | `1e-4` | 16 | `540.6` (`73.6`) | `509.1` | `8.9e-4` |
| repaired `t = 20` | Gram cone | `1e-3` | 16 | `451.5` (`108.9`) | `509.1` | `1.3e-3` |
| exact `t = 20` | either | `1e-4` | 16 | `125.8` (`3.6`) | `127.4` | `2.3e-15` |
| exact `t = 20` | either | `1e-3` | 16 | `130.3` (`21.5`) | `127.4` | `2.3e-15` |

All 128 members fire. The perturbed exact states are already inside both
sets (retained Gram `+3.8e-3`), so their projections move nothing. Their
events (`125.8`, `130.3`) match section 9's unprojected `128.2`. The
repaired states, projected, fire at `420` to `541`, against section 9's
`588.7` (64 members, unprojected). The comparison does not isolate a
projection effect:

- the projected ensembles have 16 members, against section 9's 64;
- the exact stored states and the perturbed moment vectors have different
  provenance.

Directly: all 128 tested members fire, and the projected unperturbed starts
fire at `414.8` and `509.1` (repaired) and `127.4` (exact). In these tests,
satisfying the tested necessary constraints at the start did not prevent
escape. That is not a proof that escape is independent of starting-state
geometry.

## 13.11 Cost, one session (the timing run)

**Protocol.** One trajectory per preparation, `t = 0` to 20, RK4 at
`h = 0.0025`, the map after every step, no diagnostics recorded
(`timing_family.py`). The variants were interleaved over three rounds, and
the minimum per variant is reported (all three walls in
`timing_family.json`). The host's one-minute load average was `3.7` to
`4.3` at the start of each round, on 14 cores. The cheap repair was then run
three times in the same session (`run_timing.sh`) with the recorded chain's
flags: Hartree--Fock preparation, exact-moment lift, 27 keys, Dykstra 120,
the Eq. (22) hyperplane at the `drive_power` reference, SSPRK(3,3) at
`0.0025`, `t = 0` to `0.4` (160 steps, 10 lift re-selections).

| variant | Hartree--Fock, s per unit (ratio to bare) | exact correlated, s per unit (ratio to bare) |
|---|---|---|
| `none` (bare RK4) | `0.140` (`1.00`) | `0.139` (`1.00`) |
| `minors` | `0.165` (`1.18`) | `0.164` (`1.18`) |
| `minors_energy` | `0.177` (`1.27`) | `0.174` (`1.26`) |
| `minors_exact` | `0.216` (`1.55`) | `0.426` (`3.07`) |
| `gram` | `0.165` (`1.18`) | `0.162` (`1.17`) |
| `gram_exact` | `0.197` (`1.41`) | `0.198` (`1.43`) |
| `boson_minors_exact` | `0.224` (`1.60`) | `0.417` (`3.01`) |

The cheap repair in the same session, `t = 0` to `0.4`, took `138.7`,
`138.8` and `139.2` s (`cheap_repair_timing_run{1,2,3}/summary.json`). The
minimum gives `346.8` s per unit, made up of:

- `117.2` s for the 10 lift re-selections (8 recorded as
  `lift_reselection_failures`);
- `19.0` s for the 480 Dykstra repairs (one per SSPRK stage; receipt
  `cheap_repair_timing_run1/summary.json`; this line printed the second
  run's `18.8` before the 2026-10-18 revision);
- `2.5` s for the velocity evaluations.

The recorded chain's wall over `t = 0` to 60 is `12,990.087` s, or `216.5` s
per unit. That run was in another session, over a different horizon.

These timings measure the protocol as run, including 8 of 10 failed lift
re-selections. They are not the cost of a successfully accepted repaired
trajectory. Minimum-of-three walls, a historical 60-unit chain and
different solvers and horizons are descriptive comparisons, not
matched-accuracy or general resource guarantees.

Ratios, one Hartree--Fock trajectory each, with both walls:

| map | against this session's `346.8` s per unit | against the recorded `216.5` s per unit |
|---|---:|---:|
| minors map | `2.1e3` times | `1.3e3` times |
| exact energy hold | `1.6e3` times | `1.0e3` times |
| bare RK4 step | `2.5e3` times | `1.55e3` times |

The Dykstra repairs alone (`46.9` s per unit) cost `284` times the minors
map. Section 11's `0.20` against `0.46` came from separate runs, and so did
R3's `1.08`. This session's `1.18` replaces both.

## 13.12 What remains unmeasured

- **Long horizons with exact energy.** Every exact-energy run stops at 600
  units. The minors map without the hold keeps heating to 3000 units
  (13.9). Whether the capped hold keeps the sampled coordinates bounded
  beyond 600 is not measured.
- **The positive extension.** No map of this round holds the 36-row scaled
  extension, and none is scored against it. Its role in the cheap repair's
  accuracy is not separated from the retained Gram's or the hyperplane's.
- **A converged map onto the intersection** of the Gram cone and the energy
  surface. `gram_exact` is one pass of each and is not `h`-converged.
  Alternating them to convergence is not run.
- **The repaired chain with the repair switched off** at `t = 20` and
  scored against the exact trajectory is not run. Section 9 and 13.10
  give only its escape time.
- **Other repaired states.** Every repaired-chain result rests on the
  states at `t = 20`, 40 and 60 of one chain. Task 11 is the only other
  chain, and its exact correlated cheap-repair accuracy is scored only to
  `t = 20`.
- **Ensembles.** The accuracy table is one unperturbed trajectory per
  preparation. The confinement cells are 8 members, and the triggered guard
  16. No interval is attached to a zero-of-8 count. With 0 of 8 fired, the
  exact 95 percent upper bound on the per-member probability of an event
  within the span is `0.31` (`0.17` for 0 of 16). The triggered guard's
  accuracy is scored from Hartree--Fock only.
- **The guard's thresholds** (13.8) were chosen on the ensembles they are
  tested on. A held-out test is not run.
- **Convergence in the batched drop-`trace` cell.** `maps2` does not count
  non-converged map steps. For the exact-correlated drop-`trace` cell
  (13.5) only the energy residual (at most `1.0e-12`) is recorded, not the
  remaining bound violation.
- **Execution provenance.** `run_spec.py` restores a checkpoint without
  validating saved source, input or settings hashes (its lines 62 to 68).
  It computes the state-artifact hash after execution (line 112), and the
  manifest holds no `paper_5/src` file. The final hashes prove artifact
  integrity, not fixed source, input and settings across resumes. These
  outputs are protocol-qualified exploratory observations.
- **The whole-increment gate.** The 19.5-minute process watchdog and the
  batched reruns kept each process under 20 minutes. They do not establish
  the 1200-second bound for the complete increment (preparation,
  reference, scoring, report, review and closure), which is unestablished.
- **The prerequisite holds stand.** Nothing here repairs the three current
  strict resume-parity failures (first divergence at `t = 0.1225`, about
  `1.94e-4` against `1e-12`, as root records them) or changes root 025.
  None of the studies listed in this section is selected.

## 13.13 Reproduction and hashes

Every script and receipt of this round is under
`output/local_runs/paper_v_instability_20261010/task2_corrections/`. The
interpreter is `/Users/lordskyler/Holstein_test/.venv/bin/python`, with
`PYTHONPATH=paper_5/src` and the worktree as cwd.

- **Gates:** `gate_family.py`, `gate_maps2.py` and `gate_drop_trace.py`
  check that the batched and vectorized drivers reproduce the reference
  drivers. Differences: `0.0`.
- **Launchers:**
  - `run_ablation.sh` and `run_keep.sh`: 13.5;
  - `run_obs_accuracy.sh`: 13.6, followed by `cheap_repair_accuracy.py` and
    `render_obs_table.py`;
  - `run_specs.sh` and `run_specs_b2.sh`: 13.7 to 13.9;
  - `run_boson_ec.sh` and `run_drop_trace_ec.sh`: the batched reruns;
  - `representable_escape.py`: 13.10;
  - `run_timing.sh`: 13.11;
  - `section9_recheck.py`: 13.1;
  - `order_check.py`: 13.4.
- **Watchdog:** `watchdog.sh` stops any process of this round at 19.5
  minutes and records what it stopped in `watchdog.log`. It stopped one
  run: the single-process exact-correlated `boson_minors_exact`, at 19:38,
  rerun in 150-unit batches. The single-process exact-correlated
  drop-`trace` cell was stopped twice by the same 19.5-minute guard in the
  waiting loop, and rerun through `maps2` in 100-unit batches. No process
  of this round ran 20 minutes. That is a per-process statement; it does
  not establish the 1200-second whole-increment gate (13.12).
- **Hashes:** `sha256_task2.json` and its rendering `sha256_task2_table.md`
  now cover sections 8 to 13. That includes `clip_ens_*.log`,
  `clip_accuracy_*.log`, `escape2.log`, `start_states_table.json`, every
  `task2_corrections/*.{json,npz,log,py,sh,md}`, and the three same-session
  cheap-repair segments: 444 files, 279 of them under `task2_corrections/`.
  The table renders 305 rows; it leaves out the per-member series, the
  `resume.npz` files and the scripts, which stay in the JSON manifest.
  All 78 rows of the section-12 table carry the same digest in the new
  manifest, so no earlier receipt changed. The manifest file itself has
  SHA-256 `777ae638beb3707da344c47cd2b9ca7936e2b8ee580aa98e41940f8ac0615da0`.
  These hashes bind the current bytes; they do not prove fixed source,
  input or settings across resumes (13.12).

---

# 14. The guard at every step with the cheap repair at sparse intervals (2026-10-11)

**Branch `claude/paper-v-routes-20260914`, HEAD `5d0bd578` (section 13 committed
at `36b686b3`). Exploratory, not promoted. No module under `paper_5/src`
edited, no default changed, nothing committed by this round.** Receipts are
under `output/local_runs/paper_v_instability_20261010/task3_hybrid/`, and
their SHA-256 are in the rehashed `sha256_task2.json` (14.8). This lane is not
the repair-early-then-bare lane: that lane repairs continuously over
`[0, t_h]` and then runs bare, while here the guard acts after every step and
the repair acts once every `Delta` units from `t = 0` to 60.

## 14.0 The answer first

1. **No `Delta` brings the hybrid near the cheap repair's accuracy.** From
   Hartree--Fock the best arm is `Delta = 1`. Against the repaired chain
   over `[0, 60]` it is:
   - `1.56` times worse on the site-0 occupation (`0.2506` against
     `0.1604`);
   - `24` times worse on the Eq. (22) energy (`0.0289` against `0.0012`);
   - `1.16` times worse on all 31 coordinates (`0.2919` against `0.2523`).

   Over `[0, 4]` the same arm is `4.6` (receipt values; this line printed
   `4.5` before the 2026-10-18 revision) and `26` times worse on the site-0
   occupation and the energy.
2. **From the exact correlated preparation the effect of the sparse repair
   is metric-dependent and not monotone in `Delta`.** Before the 2026-10-18
   revision this item said the sparse repair does not improve on the guard,
   which the table in 14.3 contradicts. At `T = 60`, `Delta = 5`, 10 and 20
   are better than the guard alone on site-0 and all-31 RMS at equal
   Eq. (22) RMS (`Delta = 10`: `0.2131` and `0.1975` against `0.3004` and
   `0.2513`). `Delta = 1` and `2` are worse:
   - site-0 RMS `0.3788` and `0.3710`, against `0.3004`;
   - all-coordinate RMS `0.3281` and `0.3240`, against `0.2513`;
   - Eq. (22) RMS `0.0610` at `Delta = 1` and `0.0412` at `Delta = 2`,
     against `0.0354`.

   One cell beats the reference: `Delta = 5` has site-0 RMS `0.0575` over
   `[0, 20]`, against task 11's `0.0681`. The neighbouring arms give
   `0.0918` (`Delta = 2`) and `0.1200` (`Delta = 10`), so this is not a
   trend.
3. **Over the guard alone, from Hartree--Fock, the `T = 20` and `T = 60`
   scores improve steadily as `Delta` shrinks.** `Delta = 1` lowers the
   `T = 60` scores by 13 percent (site-0), 42 percent (Eq. (22)) and 20
   percent (all 31). It costs `3.26` s per unit against the guard's `0.22`.
   - At `T = 4` the improvement is metric-dependent. `Delta = 1` improves
     the site-0 occupation (`0.0318` against `0.0356`) and the energy
     (`0.0277` against `0.0437`), but worsens all-31 RMS (`0.1364` against
     `0.1313`). `Delta = 2` slightly improves the energy (`0.0432`) and
     worsens the other two.
   - No tested arm improves all three `T = 4` scores.
   - These are single unperturbed trajectories, and only five values of
     `Delta` were tested.
4. **Cost (14.4).** The hybrid costs `0.36` to `3.26` s per unit from
   Hartree--Fock and `0.53` to `3.08` from exact correlated. The cheap
   repair costs `332.1` s per unit in the same session. So the hybrid is
   100 to 930 times cheaper.
   - Averaged per arm, the lift re-selection takes `2.4` to `3.0` s of a
     mean event wall of `2.5` to `3.1` s. It returned "infeasible" in all
     222 main-arm events and in the 12 events of the `h/2` arm (234 in all),
     so the carried lift was kept every time.
   - With the re-selection skipped, the same repair inputs would cost about
     `0.22` to `0.29` s per unit (exact correlated `Delta = 20`: `0.594`).
     This is derived, not measured. The refused branch returns a copy of the
     carried lift, but solver and RNG side effects were not audited, and no
     skipped-selection run checked bitwise equality (14.4).
   - The cheap repair's `332.1` s per unit includes 8 of 10 failed lift
     re-selections. The ratios compare the measured protocols, not equally
     accurate, successfully accepted methods.
5. **Escape (14.2, 14.3).** The hand-on state's lifetime under the bare
   closure does not order with `Delta`.
   - From Hartree--Fock, 0 of 8 fire by 600 at `Delta = 1`, 2 and 5. At
     `Delta = 10` all 8 fire (mean `481.4`), and at `Delta = 20` all 8 fire
     at `227.9`. The guard alone gives 8 of 8 (`368.8`). The repaired chain,
     same seeds, gives 8 of 8 (`373.6`; `410.0` over all 64 members to 1500).
   - From exact correlated, 0 of 8 fire at `Delta = 5`, 10 and 20; 3 of 8 at
     `Delta = 1`; 4 of 8 at `Delta = 2`; 6 of 8 for the guard alone.
   - These are finite, censored observations (8 members, 600 units). With 0
     of 8 fired, the exact 95 percent upper bound on the per-member event
     probability is `0.31`. All 117 escape integrations ended with a
     nonfailure solver status: 76 reached 600 and 41 stopped at the
     terminal event.
6. **The retained Gram is not held, even at the repair times.**
   - Every recorded post-repair, post-guard Gram minimum in the 234 events
     is negative (largest `-7.75e-6`, smallest `-0.196`).
   - Only 99 of the 234 events leave it above its pre-event value: some
     repairs reduce the violation, and most do not.
   - Every recorded extension minimum after a repair is negative too (best
     `-1.85e-4`).
   - From Hartree--Fock at `Delta = 1` the retained Gram falls to `-0.20`
     and is below `-1e-4` at 7.4 percent of grid samples. The guard alone is
     below at 75 percent, and the repaired chain at 24 percent (minimum
     `-8.5e-4`).

   These are results of the capped fallback hybrid (one Dykstra-120 pass,
   with a carried lift after every infeasible re-selection), not of an
   accepted repair.

## 14.1 What one repair step is on the closure

The flow is the bare closure (`closed_scalar_rhs`), RK4 at `h = 0.0025`,
with the exact-energy minors guard after every step (`maps2` `minors_exact`,
section 13.6). At `t = Delta, 2 Delta, ..., 60` the recorded repair step is
applied once (`hybrid.repair_event`):

1. **The closure state in the repair's coordinates.**
   - The 29 raw moments come from the exact change of variables. The guard
     holds the shared trace at 0, which that change requires.
   - The repair also needs the 27 dictionary moments, which the closure
     does not carry. They are set by the zero-cumulant rule from the
     closure's own second-order moments
     (`moment_hierarchy._zero_terminal_cumulant_moment`): two-spin moments
     are products of one-spin means, and third-order moments have zero
     third cumulant.
   - This is an assumption of this lane. It equals the exact-moment
     dictionary of the Hartree--Fock product state to `4.4e-16`. It differs
     from the exact correlated ground state's dictionary by up to `0.378`
     (`probe_hybrid_*.json`). Agreement at the initial product state does
     not establish correctness at later hybrid states.
2. **The lift.** The lift is re-selected by the recorded rule: the declared
   proximal log-det completion, warm-started from the carried lift, with
   measured candidates and acceptance margin `1e-3`. A refusal keeps the
   carried lift. The carried lift starts as the exact-moment lift of the
   preparation's state vector, as in the repaired chain.
3. **One repair call.** `AffineGramCoordinates.repair` with the recorded
   settings: Dykstra, 120 iterations, both cones, the Eq. (22) hyperplane.
   The energy reference is the computed Eq. (22) energy carried into the
   repair (`hybrid.repair_event`). The closure step changes the energy only
   by the drive's work plus integration error, and the guard holds it. So
   this approximates the chain's drive-power reference; no separate
   finite-step drive-work accumulator was kept to show that it is the same
   protocol.
4. **Back to the closure.** The repaired raw moments are mapped back to the
   31 coordinates, the dictionary is dropped, and the guard is applied once.

Every other setting is the repaired chain's (`run_divergence_segment.py`
flags: 27 keys, `--lift exact_moments`). Checks (`probe_hybrid.py`):

- the closed start equals the stored bare start to `1.0e-10`
  (Hartree--Fock) and `8.0e-16` (exact correlated);
- the raw round trip is exact to `5.6e-17`;
- the route's Eq. (22) functional equals `maps2.energy` to `4.4e-16`;
- across all 234 events (222 main-arm, 12 in the `h/2` arm), the energy
  after the repair and the guard differs from the reference by at most
  `3.6e-15`.

Accuracy columns give the site-0 occupation / Eq. (22) energy / all 31
coordinates. Each is the RMS over every `0.0025`-grid sample in `[0, T]`
against the exact driven trajectory from the same preparation
(`_accuracy_metrics`; references as in 13.6). "s per unit" is the arm's own
wall in the serial session, divided by 60. It counts guard segments and
repair events only, not the route build, checkpoints, the escape test,
scoring or the report.

The Gram columns use the retained 8-by-8 joint Gram's minimum eigenvalue,
taken over all 24,001 grid samples in `[0, 60]`. At a repair time the sample
is the state after the repair.

The escape column is the bare-closure test from the arm's `t = 60` state:
8 members `x60 + 1e-4 u` (`u` the admissible direction, seeds 1 to 8),
DOP853 `rtol 1e-10`, `atol 1e-12`, event `max|x_i| = 1e4`, span 600. It
gives:

- the number fired by 600;
- the restricted mean to 600 (the mean of `min(event, 600)`, a lower bound
  on the mean when members are censored);
- the unperturbed state's own event time.

## 14.2 Hartree--Fock preparation

Bare RK4 step `0.130` s per unit, guard `0.203` (`timing_bare_guard.json`,
same session).

| `Delta` | T = 4 | T = 20 | T = 60 | s per unit | Gram minimum | fraction below `-1e-4` | escape: fired by 600 of 8; restricted mean; unperturbed |
|---|---|---|---|---:|---:|---:|---|
| 1 | 0.0318 / 0.0277 / 0.1364 | 0.1420 / 0.0288 / 0.2695 | 0.2506 / 0.0289 / 0.2919 | 3.258 | -0.200 | 0.074 | 0; 600.0; none |
| 2 | 0.0370 / 0.0432 / 0.1392 | 0.1528 / 0.0477 / 0.2906 | 0.2600 / 0.0484 / 0.3063 | 1.711 | -0.354 | 0.216 | 0; 600.0; none |
| 5 | 0.0356 / 0.0437 / 0.1313 | 0.1603 / 0.0487 / 0.2966 | 0.2666 / 0.0495 / 0.3211 | 0.778 | -0.381 | 0.400 | 0; 600.0; none |
| 10 | 0.0356 / 0.0437 / 0.1313 | 0.1638 / 0.0487 / 0.3020 | 0.2782 / 0.0495 / 0.3335 | 0.480 | -0.381 | 0.809 | 8; 481.4; none |
| 20 | 0.0356 / 0.0437 / 0.1313 | 0.1655 / 0.0487 / 0.3146 | 0.2852 / 0.0495 / 0.3570 | 0.357 | -0.381 | 0.863 | 8; 227.9; 223.0 |
| infinity (guard alone) | 0.0356 / 0.0437 / 0.1313 | 0.1655 / 0.0487 / 0.3146 | 0.2890 / 0.0495 / 0.3668 | 0.224 | -0.381 | 0.750 | 8; 368.8; 405.4 |
| every stage (repaired chain) | 0.0070 / 0.0011 / 0.0108 | 0.0736 / 0.0012 / 0.1537 | 0.1604 / 0.0012 / 0.2523 | 332.1 (this session, `t = 0` to `0.4`); 216.5 (recorded, `t = 0` to 60) | -0.00085 | 0.242 | 8; 373.6; 350.8 |

Notes on the table:

- **No repair before the first repair time.** Arms with `Delta >= 5` equal
  the guard alone over `[0, 4]`, and `Delta = 20` equals it over `[0, 20]`.
- **The guard-alone row reproduces section 13.6's `minors_exact` row** to
  every printed digit.
- **The repaired-chain row:**
  - its accuracy is `cheap_repair_accuracy.json`;
  - its Gram is computed the same way on its stored coordinates;
  - its escape is section 9's receipt for seeds 1 to 8 (all fire within
    1500, so the count by 600 is exact).

## 14.3 Exact correlated preparation

Bare RK4 step `0.132` s per unit, guard `0.408` (same session).

| `Delta` | T = 4 | T = 20 | T = 60 | s per unit | Gram minimum | fraction below `-1e-4` | escape: fired by 600 of 8; restricted mean; unperturbed |
|---|---|---|---|---:|---:|---:|---|
| 1 | 0.0225 / 0.0532 / 0.1289 | 0.0927 / 0.0599 / 0.1579 | 0.3788 / 0.0610 / 0.3281 | 3.081 | -0.197 | 0.609 | 3; 584.6; none |
| 2 | 0.0280 / 0.0377 / 0.0900 | 0.0918 / 0.0407 / 0.1612 | 0.3710 / 0.0412 / 0.3240 | 1.623 | -0.195 | 0.889 | 4; 512.2; 585.9 |
| 5 | 0.0192 / 0.0352 / 0.1059 | 0.0575 / 0.0353 / 0.1047 | 0.2857 / 0.0354 / 0.2441 | 0.804 | -0.433 | 0.414 | 0; 600.0; none |
| 10 | 0.0192 / 0.0352 / 0.1059 | 0.1200 / 0.0353 / 0.1463 | 0.2131 / 0.0354 / 0.1975 | 0.529 | -0.749 | 0.997 | 0; 600.0; none |
| 20 | 0.0192 / 0.0352 / 0.1059 | 0.1347 / 0.0353 / 0.1537 | 0.2725 / 0.0354 / 0.2459 | 0.714 | -0.749 | 0.986 | 0; 600.0; none |
| infinity (guard alone) | 0.0192 / 0.0352 / 0.1059 | 0.1347 / 0.0353 / 0.1537 | 0.3004 / 0.0354 / 0.2513 | 0.353 | -0.749 | 0.974 | 6; 504.6; 478.5 |
| every stage (task 11, run to `t = 20`) | 0.0116 / 0.0040 / 0.0093 | 0.0681 / 0.0053 / 0.0560 | not run | not measured | --- | --- | no `t = 60` state |

Notes on the table:

- **The reference is not the same driver.** No every-stage repair from this
  preparation exists to `t = 60` with the repaired chain's driver. The
  nearest is task 11: the same 27 keys, Dykstra 120 and hyperplane, but the
  declared-completion lift with the retry ladder, run only to `t = 20`.
- **The `Delta = 20` wall** is higher than `Delta = 10`'s. Its guard
  alternation reached the 200-iteration cap (`0.59` s per unit of guard
  alone in that arm). `maps2.run` records no bound residual, so there is no
  receipt of the bounds at those steps.
- **The guard-alone row reproduces section 13.6.**
- **The pulse work.** Repairs during the pulse (`Delta = 1`, 2) change the
  state, and so can change the work the drive does on the closure. The
  Eq. (22) RMS rises above the guard's (`0.0610` and `0.0412` against
  `0.0354`). The attribution to pulse work is not separately measured.

## 14.4 Cost, one session

The arms ran serially in one session (`run_arms.sh`, `session.log`,
13:58 to 14:16; one-minute load average `2.2` to `6.0`, mostly `3.0` to
`4.3`, on 14 cores). It does not establish the 1200-second whole-increment
gate: the escape test, scoring, the report and its review ran afterwards,
and no complete-increment receipt exists. The session ran in this order:

1. the bare step and the guard (`timing_bare_guard.py`: one trajectory per
   preparation, `t = 0` to 20, three interleaved rounds, minimum);
2. the 12 arms and the `h/2` arm, each its own process;
3. the cheap repair from Hartree--Fock, `t = 0` to `0.4`, with the
   recorded chain's flags. It took `132.8` s, so `332.1` s per unit:
   `112.5` s of 10 lift re-selections (8 recorded as
   `lift_reselection_failures`), `17.9` s of Dykstra repairs and `2.4` s of
   velocity evaluations. That is the protocol's cost including those
   failures, not the cost of an accepted repaired trajectory.

| `Delta` | Hartree--Fock: s per unit (repair events; mean per event) | exact correlated: s per unit (repair events; mean per event) | with the re-selection skipped (derived), Hartree--Fock / exact correlated |
|---|---|---|---|
| 1 | `3.258` (60; `3.05` s) | `3.081` (60; `2.83` s) | `0.249` / `0.288` |
| 2 | `1.711` (30; `3.00` s) | `1.623` (30; `2.72` s) | `0.229` / `0.283` |
| 5 | `0.778` (12; `2.85` s) | `0.804` (12; `2.74` s) | `0.217` / `0.264` |
| 10 | `0.480` (6; `2.67` s) | `0.529` (6; `2.53` s) | `0.218` / `0.281` |
| 20 | `0.357` (3; `2.69` s) | `0.714` (3; `2.46` s) | `0.224` / `0.594` |
| infinity | `0.224` (0) | `0.353` (0) | --- |

Each event's wall splits as follows:

- the re-selection takes `2.4` to `3.0` s;
- the repair call takes `0.039` to `0.044` s;
- the coordinate maps and the guard take under `1e-3` s.

Every re-selection returned "infeasible": 222 of 222 main-arm events
across both preparations, and 12 of 12 in the `h/2` arm. The solved lift
failed the measured-candidate check, and the refused branch returns a copy
of the carried lift, so the repair received the carried lift unchanged.
That is source-level support for equal repair inputs if the re-selection
were skipped. Solver and RNG side effects were not audited, and no
skipped-selection run checked bitwise trajectory equality. The last column
is the arm's wall minus its re-selection seconds: derived from the same
run, not measured.

Ratios at `Delta = 1` from Hartree--Fock:

| compared with | ratio |
|---|---:|
| the cheap repair, this session | `1/102` |
| the cheap repair, recorded wall | `1/66` |
| the guard | `14.5` times |
| the guard, with the re-selection skipped | `1.11` times |

## 14.5 What a repair event does

Per event (`arm_*.json`, `records`):

| quantity | Hartree--Fock | exact correlated |
|---|---|---|
| largest coordinate change of the closed state, median over arms | `0.04` to `0.16` | `0.06` to `0.22` |
| largest coordinate change of the closed state, maximum | `0.31` | `0.35` |
| scaled-extension minimum after the repair | `-1.9e-4` to `-4.1e-3` (both preparations) | |
| retained Gram, before the event | down to `-0.33` | down to `-0.37` |
| retained Gram, after the repair and the guard | `-3e-4` to `-9e-3` | down to `-0.20` |

Dykstra at 120 iterations does not reach the extension cone: every recorded
extension minimum after a repair is negative (best `-1.85e-4`). The
retained Gram is not held at the repair times either. Every post-repair,
post-guard minimum is negative (largest `-7.75e-6`), and 99 of the 234
events leave it above its pre-event value. From Hartree--Fock the guard
applied after the repair brings the Gram near 0 (`-3e-4` to `-9e-3`). From
exact correlated it does not.

## 14.6 Step dependence

The Hartree--Fock `Delta = 5` arm was rerun at `h/2 = 0.00125`, with repairs
at the same times:

- the largest score change at `T = 4`, 20 and 60 is `1.8e-4`;
- the largest coordinate difference is `3.8e-4`, `7.9e-4` and `8.8e-4` at
  `t = 4`, 20 and 60;
- the escape test gives the same 0 of 8.

This is the observed difference for this one arm. It does not show that
the whole table is converged, or certify a continuum limit. The other arms
were run at `h` only.

## 14.7 What remains unmeasured

- **The dictionary rule.** Only the zero-cumulant rule was run. Other rules,
  such as carrying the repair's own dictionary between events or using
  exact third moments where available, may change every row from the exact
  correlated preparation.
- **`Delta` below 1.** From Hartree--Fock the `T = 20` and `T = 60` scores
  improve monotonically down to `Delta = 1`; the `T = 4` scores are
  metric-dependent. Whether they keep improving at `0.5` or `0.25` units
  is not measured. With the re-selection skipped those arms would cost
  about `0.29` and `0.37` s per unit.
- **The hybrid with the re-selection skipped, run directly.** Its cost is
  derived above. Equal repair inputs have source-level support (the
  refused branch returns the carried lift). Solver and RNG side effects
  were not audited, and no run checks bitwise equality.
- **A converged repair.** At 120 iterations the extension minimum stays
  negative after every event.
- **Perturbed members.** The accuracy is one unperturbed trajectory per
  arm.
- **Escape statistics.** The escape test uses 8 members to 600 units.
  Restricted means are lower bounds wherever members are censored, and no
  interval is attached. With 0 of 8 fired, the exact 95 percent upper bound
  on the per-member probability of an event within 600 is `0.31`.
- **The exact correlated every-stage reference.** No run with the repaired
  chain's driver goes to `t = 60` from this preparation, so the T = 60
  column has no reference.
- **`h/2` for the other arms**, and for the exact correlated preparation.
- **The whole-increment gate.** Per-process timers and the serial session
  do not establish the 1200-second whole-increment bound, which is
  unestablished.
- **The prerequisite holds stand.** Nothing here repairs the three current
  strict resume-parity failures or changes root 025. None of the studies
  listed in this section, or in the consolidated results' unmeasured list,
  is selected.

## 14.8 Reproduction and hashes

Everything is under
`output/local_runs/paper_v_instability_20261010/task3_hybrid/`. The
interpreter is `/Users/lordskyler/Holstein_test/.venv/bin/python`, with
`PYTHONPATH=paper_5/src` and the worktree as cwd.

- `hybrid.py`: the construction of 14.1.
- `probe_hybrid.py`: the checks of 14.1.
- `run_arm.py`: one arm. It is resumable from `arm_<tag>_ck.npz` and stops
  cleanly at a segment boundary after 18 minutes. All 13 arms (12 main, one
  at `h/2`) reached `t = 60` with no resume (`resumes = 0` in every
  `arm_*.json`). Had a resume occurred, `run_arm.py` would have restored the
  checkpoint without validating saved inputs. The longest arm took 3.3
  minutes. All 117 escape integrations ended with a nonfailure solver
  status.
- `run_arms.sh`: the serial session.
- `escape_end.py`: the escape test.
- `score_arms.py`: writes `hybrid_table.json` and `hybrid_table.md`.
- `watchdog.sh`: stops any process at 19.5 minutes. It stopped none. A
  per-process limit does not establish the whole-increment gate (14.7).

`sha256_task2.py` now also covers `task3_hybrid/` (every `.json`, `.npz`,
`.log`, `.py`, `.sh`, `.md`, and the cheap-repair segment's `summary.json`
and `trajectory.npz`). It also now covers `spectra.py` and the three repair-driver
scripts the hybrid imports (`run_divergence_segment.py`, `exact_lift.py`,
`hartree_fock_preparation.py`). The manifest has 520 files, 72 of them under
`task3_hybrid/`. The 78 rows of the section-12 table keep their digests. The
manifest file itself has SHA-256
`767f6eec4c5facf11dc9948d88c3a9ad13907db8787bbbe98304ab88918d795f`, and its
rendering `sha256_task2_table.md` has 369 rows.
