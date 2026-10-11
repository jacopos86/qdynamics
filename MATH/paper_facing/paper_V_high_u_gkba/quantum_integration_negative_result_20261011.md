# Quantum integration of the archive equations: the negative result on the L = 2 dimer (2026-10-11)

**Status.**

- **Exploratory, not promoted.** No number here is adopted evidence.
- **Manuscript insertion is the user's decision.** That includes dropping the
  draft's cubic term (section 1). The manuscript
  `MATH/paper_details/paper_V_high_u_gkba.tex` is not edited.
- **This file consolidates; it supersedes nothing.** It contains no new runs.
  Every number is quoted from the three sources below, which stand as
  written, and each section cites them by section and commit.

| short name | file | commit |
|---|---|---|
| **feasibility** | `MATH/paper_facing/paper_V_high_u_gkba/quantum_integration_feasibility_20261010.md` (Tasks 1 to 3, and "Corrections after verification (2026-10-11)") | `a23ff642`; the corrections section in `475c47a7` |
| **r2** | `agent_guidance/paper-v/handoffs/claude-architecture-quantum-integration-20261010-r2.md` (corrections and eight calculations after four refuters and a critic) | `475c47a7` |
| **pricing** | `agent_guidance/paper-v/handoffs/claude-quantum-route-a-pricing-20261011.md` (route (a) priced end to end, with route (b) beside it) | `d3c80e8a` |

The closure-instability report
(`agent_guidance/paper-v/overnight/closure-instability-20261010.md`, commit
`40771435`) supplies the escape times and the stacked conditioning.

**Fixed configuration throughout.**

- **The dimer:** `t_hop = 1`, `gamma = omega_ph = 0.5`, `lambda_ep = 1.5`
  (`g = 0.6124`), drive amplitude 1, pulse width 1. The drive is the Gaussian
  `f(t) = 2 sin(pi t/4) exp(-t^2/2)`, with maximum `0.8664` at `t = 0.908`.
- **Targets.**
  - **T-bare:** the bare archive closure `closed_scalar_rhs` integrated from
    the exact correlated preparation (the cutoff-16 ground state), DOP853 at
    `rtol 1e-12`.
  - **T-physical:** the exact cutoff-16 driven dimer.
- **Error norms.**
  - "Max coordinate error" is the largest absolute difference over the 31
    closed coordinates.
  - "Observable error" is the absolute difference of the site-0 occupation
    per spin, `(1 + x_0)/2`, or of the Eq. (22) energy.

---

## 1. What the archive equations are, as an ODE

**Sources:** feasibility (a) and Task 2(a); r2 section 2.2; feasibility
Corrections.

**The coordinates.**

- There are 31 real closed coordinates, in blocks `rho` (3), `B` (4), `N`
  (4), `A` (6), `C` (14).
- In the draft's split, Eqs. (14a)-(14d) are the quantum target (27
  coordinates) and Eq. (14e) is the classical coherent-field update (the 4
  `B` coordinates).

**The right-hand side is exactly quadratic** (feasibility (a)):

```text
dx/dt = F_0 + (A_1 + f(t) D) x + A_2 (x (x) x)
```

- **How it was obtained.** Exact interpolation of the committed right-hand
  side on 5,984 integer points. It reconstructs that right-hand side to
  `1.7e-14` relative on 1,500 fresh random states (coordinates of standard
  deviation 0.1, 1 and 3; times in `[0, 8]`).
- **No cubic term.** `A_3` is zero to `1.5e-16`. The Eq. (14d) normal
  particle and hole sources sum to `-i N [g^T, rho]`, which cancels every
  `rho g rho` product.
- **The cubic term is zero in the draft's own coordinates too.** In the
  draft's real-doubled vectorization (44 reals including `B`), `A_3` is at
  most `7.4e-17` and the constant term vanishes exactly. `F_0` is nonzero
  only because the closed chart is affine (`rho = I/2 + ...`).
- **Coefficient counts:**

  | | `F_0` | `A_1` | `D` | `A_2` |
  |---|---|---|---|---|
  | nonzero entries | 5 | 83 | 10 | 69 monomials, only in the `rho` and `C` equations |
  | spectral norm | `0.906` | `4.047` | `1` | `1.995` |

  The `A_2` norm is that of the symmetric `31 x 961` matricization.
- **Forcing and drive.** The forcing is `F_0` (constant). The drive enters
  only multiplicatively, as `f(t) D x`, with `D` skew-symmetric (eigenvalues
  `+-i` and `0`).
- **Decision for the user.** The draft carries a cubic term,
  `C(x,x,x)` in Eq. (`eq:paper_v_force_polynomial`) and `A_3 x^{(x)3}` in the
  fault-tolerant lifting layer. It is identically zero for these equations.
  Dropping it is the user's decision.

**The linear part is skew-adjoint in a metric** (feasibility (b); r2
correction 20).

- `A_1` has all 31 eigenvalues with `|Re lambda| <= 2.2e-16`.
- It is semisimple, with eigenvector condition number `55.4`.
- It is skew-adjoint in every member of a 46-dimensional family of positive
  metrics. The family is not unique; the task-1 eigenvector metric has
  condition number 3,066.
- In the Euclidean metric of the closed coordinates it is not skew: its
  logarithmic norm is `+1.598`.

**The divergence is zero** (feasibility (b); r2 correction 28).

- `tr J(t, x) = 0` at every state and time, for the drive-free and driven
  field alike.
- More strongly, every diagonal entry `dF_i/dx_i` vanishes identically.
  Central differences of the committed right-hand side give exactly `0.0`.

**The Eq. (22) energy is an exact identity** (feasibility Task 2(a)).

- The energy `H` is quadratic in the closed coordinates.
- `grad H . F = 0` as a polynomial identity on all of R^31. The coefficients
  are exactly 0 by tensor algebra, or `7.4e-17` by interpolation.
- The drive's power is exactly `grad H . (D x) = -4 Im rho_01`.

**Coefficient parity against Eqs. (14a)-(14e) is now tested** (r2 section
2.2; `paper_5/tests/test_archive_hamiltonian_parity.py`, 11 tests).

- **Hamiltonian rows.** At 81 exact driven states (`t <= 4`), the 6 rows
  that are exact Ehrenfest equations agree with the exact contracted
  derivative to `<= 1.4e-14`, at cutoffs 16 and 20.
- **B, N, A rows.** They carry only the phonon-cutoff remainder, which falls
  from `1.2e-4`, `4.3e-5`, `7.3e-4` at cutoff 16 to `1.8e-6`, `1.2e-6`,
  `1.3e-5` at cutoff 20.
- **Second transcription.** An independent hand-coding of Eqs. (14a)-(14e)
  and (15) agrees on all 31 rows to `5.9e-16`. Its extracted `F_0`, `A_1`
  and `D` are equal exactly, and `A_2` to `1.4e-17`.
- **Mutation sensitivity.** The mutation `+0.3 x_0 x_3`:
  - leaves a defect of `0.133` in each Ehrenfest row it is put into;
  - leaves a defect of `0.159` in row 21 (`correlation_0_01_real`) through
    the transcription;
  - passed every earlier test.
- **Limits.** The two transcriptions share the coordinate maps, the
  spin-degeneracy convention and the drive sign. The 11 factorized `C` rows
  are checked only against the transcription; their closure defect against
  the Hamiltonian is up to `0.258`.

---

## 2. The structural negatives

**Sources:** feasibility Task 2(a) and Corrections; r2 corrections 14 to 19
and section 2.4.

### 2.1 No constant or affine Poisson tensor, for any Hamiltonian

The test: a Poisson tensor is preserved by each of its Hamiltonian vector
fields (`L_F J = 0`), so these searches are complete within their classes.

- **Constant tensors.**
  - The full drive-free field preserves exactly one constant antisymmetric
    tensor: rank 2, the canonical pair of the decoupled centre-mode coherent
    field `B_+ = B_0 + B_1`. The linear part alone preserves a
    41-dimensional space.
  - `F(x)` spans 30 dimensions, so no constant Poisson tensor generates it,
    whatever the Hamiltonian.
- **Affine tensors** `J_0 + sum_k x_k J_k` (Lie-Poisson on any
  31-dimensional Lie algebra, with or without a constant cocycle; 14,880
  unknowns).
  - The preserved family has dimension 4:
    `span{u^v, ell u^v, R^w, E^w}`. Here
    `w = e_phonon00 + e_phonon11 + e_phonon01real` is a constant symmetry of
    the field.
  - All four have ranges in the 3-dimensional centre-mode subspace
    `(Re B_+, Im B_+, N_++)`, so their rank is at most 2.
  - `F` lies outside that subspace (relative residual at least `0.93`), so
    no affine Poisson tensor generates it.
- **With the Eq. (22) energy specifically.** An affine `J(x)` cannot produce
  38 of the 69 quadratic monomials of `F`.
- **Lie-algebra non-closure.** The span of the raw operators behind the 31
  cumulants is not closed under commutators: 92 of 595 pairs leave the span
  (`n_c = 5`, boson occupations at most 3).
- **Qualifications.**
  - The negatives above hold for constant and affine tensors **in the closed
    chart** and under affine changes of it.
  - At quadratic order the drive-free field **is** globally Hamiltonian,
    with the Eq. (22) energy, for the degree-2 Poisson tensor
    `F^w / (2 omega)`: Jacobi residual `4.3e-16`, generation residual
    `4.4e-16`. Its leaves are 2-dimensional and it has no evident
    computational use. A rank-4 bivector `F^w + u^v` is also preserved.
  - Charts that are not affine in the closed coordinates are not covered.
    Neither is a raw-moment chart, which is a quadratic change.
  - There are nine polynomial first integrals of degree at most three (four
    at degree at most two; five survive the drive), with functional rank 5:
    `ell`, `H`, the centre-mode energy `|B_0 + B_1 + 2g/omega|^2`, and two
    cubic integrals. They admit nonlinear level-set reductions that are not
    assessed.
  - All nine are invariant under translations in a 6-dimensional subspace
    of the 8 off-diagonal `C` coordinates, so no function of them bounds
    those coordinates. They are the coordinates that escape to `1e4` at
    `t = 141.4059` from the exact correlated preparation (instability
    report).

### 2.2 No metric common to `A_1` and `D`

**Sources:** r2 correction 19 and section 2.4; `test_conserved_structure.py`.

- **The candidates.** The symmetric `G` making `A_1` skew-adjoint form a
  46-dimensional space. Requiring `D` skew-adjoint too cuts it to 8
  dimensions.
- **The proof that none is positive.** The matrix Lie algebra generated by
  `A_1` and `D` has dimension 91 and contains a nonzero `X` with
  `||X^2||_F = 2.4e-13`, nilpotent of index 2. So `||exp(t X)||_2` grows
  linearly: `647` at `t = 1000`. A positive `G` making `A_1` and `D` skew
  would bound it by `sqrt(cond G)`.
- **After reduction.** The same holds on the 28-coordinate system that
  removes the decoupled centre mode and the linear first integral:
  dimensions 34 and 4, algebra dimension 91, `||exp(1000 X)|| = 679`.
- **Driven linear propagator.** It is therefore priced directly. For
  `A_1 + f(t) D` alone, `sup ||Phi(t,s)||_2` is `12.086` to `t = 20`
  (undriven `12.084`; grid maximum on 0.05 grids).

### 2.3 Nonresonance fails at order two

**Source:** r2 section 2.4.

- In the eigenbasis of the linear part there are 79 ordered resonant triples
  `lambda_c = lambda_a + lambda_b`.
- 33 of them carry nonzero quadratic coupling: smallest `0.031` on the full
  system and `0.022` on the reduced one; the largest vanishing coupling is
  `1.1e-15`.
- 29 of the 33 involve a zero eigenvalue. The counts are identical on the
  full and the 28-coordinate system.

### 2.4 The dissipative Carleman condition cannot be met

**Sources:** feasibility (b) and Verdict; r2 section 2.4.

- **The Liu et al. condition.** It needs every eigenvalue of the linear part
  to have negative real part, and `R < 1`. At the exact correlated
  preparation (`||x_0|| = 2.283`) that means `Re lambda_1 < -4.95`; at the
  Hartree-Fock preparation (`||x_0|| = 1.803`), `-4.10`. The measured value
  is `Re lambda_1 = +1.05e-16`.
- **The failure is structural.** `tr J = 0` everywhere, so no expansion
  point and no invertible linear change of coordinates makes every
  eigenvalue's real part negative.
- **The conserved-system conditions (Jennings et al.) fail too.** On the
  28-coordinate reduction:
  - the linear part's rank is 24, the forcing adds nothing (rank 24), and
    the quadratic term raises the rank to 28;
  - all 4 left zero modes receive a nonzero quadratic projection, with
    singular values `1.09`, `0.84`, `0.82`, `0.78`;
  - the trace is zero, so there is no strictly stable complement;
  - 4 modes fit neither signed sector;
  - a share `1.00` of the eigenbasis quadratic tensor lies outside the
    allowed sector blocks.
- **Scope.** These are failures of sufficient conditions, not a proof that
  no quantum algorithm exists. Sections 3 and 4 price the routes directly.

---

## 3. The route costings, side by side

All on T-bare unless labelled. **Sources:** feasibility (c) and Tasks 2 to
3; r2 sections 2.1, 2.3, 2.5, 2.7, 2.8; instability report table 5.1.

### 3.1 Carleman, origin-centred lift (the fault-tolerant lifting layer as drafted)

**Lifted dimension.** These are without the constant block.

| order | Kronecker | distinct monomials |
|---|---|---|
| 2 | 992 | 527 |
| 3 | 30,783 | 5,983 |
| 4 | 954,304 | 52,359 |

**Truncation error of the solution** (feasibility (c)): the max coordinate
error of block 1 against T-bare.

| order | `t <= 1` | `t <= 4` | `t <= 20` |
|---|---|---|---|
| 1 | `0.261` | `0.957` | `1.66` |
| 2 | `0.0513` | `0.981` | `12.3` |
| 3 | `0.0076` | `1.036` | `72.1` |
| 4 | `0.0012` | `1.001` | `387` |

The truncation converges in order only for `t <= 2`.

**The Euclidean first-block error.** It is `2.25` (order 2) and `2.05`
(order 3) at `T = 4`, and `19.6` and `127` at `T = 20` (r2 correction 2).

**The tail identity** (r2 correction 1).

- The error equals the propagated omitted tail by Duhamel's formula, exact
  by construction.
- Evaluated with an RK4 adjoint, the trapezoid rule reproduces it to
  `2.6e-6` to `5.2e-6` relative, and Simpson to `1.1e-9` to `4.1e-9`. That
  validates the pipeline; it is not a finding.

**The norm certificate is loose** (r2 correction 3).

- `int ||P_1 U_K(T,s) P_K|| ||tau_K(s)|| ds` is 22, 147, 109 and 657 times
  the error at (order, `T`) = (2, 4), (2, 20), (3, 4), (3, 20).
- That is a sampled, a-posteriori quantity. Its looseness splits into norm
  misalignment (`11.7` to `78`) and cancellation in time (`1.6` to `12.5`).
- The observable-specific bound at order 4, `T = 4` is 3.34 times the error
  (site-0 occupation) and 2.10 (energy's linear part).

**First-block weights** (r2 correction 4 and section 2.5).

- Encoding: Kronecker power, constant block excluded.
- Unscaled (`gamma = 1`): `0.261`, `0.0561`, `0.00775` at `T = 4` for orders
  2, 3, 4, and `0.182`, `0.0321`, `0.00449` at `T = 20`. The postselection
  probability is the square: `0.068`, `3.2e-3`, `6.0e-5` at `T = 4`.
- Rescaled by `||x_0|| = 2.283`: `0.525`, `0.249`, `0.080` at `T = 4`.
- Distinct-monomial encoding: `0.347` and `0.111` (orders 2, 3).

**Block normalization** (`gamma = 1`, `f` at its maximum).

| order | `s x max|entry|` | LCU bound | spectral norm |
|---|---|---|---|
| 2 | 200 | 15.6 | 6.65 |
| 3 | 1,660 | 23.4 | 9.73 |
| 4 | 3,324 | 31.3 | 12.86 |

**Observable error against T-bare at `T = 4`** (r2 section 2.1).

| order | site-0 occupation | energy read from block 2 |
|---|---|---|
| 2 | `0.290` | `0.178` |
| 3 | `0.132` | `0.029` |
| 4 | `0.0915` | `0.0094` |

The origin-centred lift never meets `1e-2` on the site-0 occupation.

**The linearized flow's amplification** along T-bare (r2 section 2.3). This
is `Lambda = sup over 0 <= s <= t <= T of ||Phi(t,s)||_2` for
`J = A_1 + f D + 2 A_2(x(t), .)`.

- Grid supremum: `12.98` (`T = 4`) and `51.04` (`T = 20`).
- With between-grid inflation and an RK4 allowance: `13.22` and `54.10`.
- Admissible time-averaged defect for a nonlinear tube: `4.48e-5` and
  `1.07e-7`.
- To `T = 60` the sampled supremum is `3,169` (instability report).

**Stacked backward-Euler conditioning** (T-bare, instability report).

| `T` | condition number | from the step count alone |
|---|---|---|
| 4 | 12,071 | 2,038 |
| 20 | 166,189 | 10,187 |
| 60 | 11,558,518 | 30,559 |

### 3.2 Carleman, lift re-centred at each read-out state (windowed)

**Sources:** r2 section 2.1; pricing sections 2 and 3.

- **The scheme.** At each restart the order-`K` truncation of `y = x - x_r`
  is built about the read-out state `x_r`, with forcing `F(t, x_r)`.
- **With free exact readout, order 3:**

  | re-lifts | window | max coordinate error, `t <= 4` / `t <= 20` |
  |---|---|---|
  | 79 | 0.25 | `4.1e-8` / `6.3e-7` |
  | 19 | 1 | `1.1e-4` / `1.8e-3` |

  At 19 re-lifts the observable errors at `T = 4` / `20` are `2.6e-5` /
  `1.6e-4` on the site-0 occupation and `8.4e-5` / `1.2e-4` on the energy.
- **The origin-centred windowed lift is far worse.** At the same order and
  79 re-lifts it reaches only `2.6e-3` / `0.30`. **Any lift must be
  re-centred.**

### 3.3 Koopman--von Neumann

**Sources:** feasibility Task 2(b) and Corrections; r2 corrections 21-22 and
section 2.7.

- **The generator.** It is `-i F . grad`, because `div F = 0`. It is
  formally Hermitian for any smooth `F`; no Hamiltonian structure is needed.
- **Register and generator norm.** The figures depend on how the grid is
  sized, at coordinate precision `epsilon = 1e-2` and box width `W = 4`:

  | rule | `T = 4` | `T = 20` | `T = 60` |
  |---|---|---|---|
  | heuristic `h = epsilon/(3 Lambda)`, qubits | 434 | 496 | 682 |
  | same, `T ||L||` with the generator norm sampled on the trajectory | `4.9e5` | `1.7e7` | `5.6e9` |
  | `T ||L||` with the generator norm over the box (a lower bound) | `>= 2.0e7` | `>= 3.9e8` | `>= 7.3e10` |
  | qubits to resolve the compressed direction | 558 | 682 | 1,023 |
  | qubits from the B-Q2 dossier's Lipschitz envelope | 651 | 1,984 | 5,518 |

  The heuristic is neither a lower bound nor a sufficient cost.
- **The readout is biased.** A packet-mean readout is biased as
  `sigma_0^2`. Keeping the bias at most `1e-2` needs a packet width
  `sigma_0 <= 0.020` at `T = 4` and `<= 0.0024` at `T = 20`. At
  `sigma_0 = 0.1`, 86 of 2,000 samples escape by `T = 20`.
- **Against the correct baseline.** The same problem, the dimer at cutoff
  16, is directly simulated on 12 qubits (11 dense) with
  `T (||H|| + max|f|) = 102`, `511`, `1,533` at `T = 4`, `20`, `60`. KvN
  therefore needs 36 to 57 times the qubits and `4.8e3` to `3.7e6` times the
  time-norm product.

### 3.4 Schrödingerization of the trajectory-linearized flow (the fault-tolerant linearization layer)

**Sources:** feasibility Task 2(c) and Corrections; r2 corrections 20 and 23
and section 2.8.

- **The recovery exponent depends on the metric.** Along T-bare, in the
  integral form `int max(mu_G, 0) dt` and the theorem form `T sup mu_G`:

  | metric | integral form, `T = 4` | integral form, `T = 20` | theorem form, `T = 4` | theorem form, `T = 20` |
  |---|---|---|---|---|
  | Euclidean | `7.11` | `37.26` | `7.38` | `46.13` |
  | constant, fitted along T-bare (`sqrt(cond G) = 8.45`) | `1.86` | `9.50` | `2.09` | `11.77` |
  | constant, fitted along the exact trajectory (`sqrt(cond G) = 7.72`) | `2.18` | `12.52` | `3.08` | `17.91` |

  Along the exact trajectory itself, the task-1 eigenvector metric gives
  `51` and `251` in the integral form, which is worse than Euclidean. A
  time-dependent metric is not needed to remove most of the exponent.
- **The recovery factor.** For the forced linearized flow, `xi(0) = 0` with
  the closure defect as source, along the exact trajectory in Euclidean
  coordinates, `e^{p} ||u(0)|| / ||u(T)||` is `1.2e3` to `1.5e3` at `T = 4`
  and `8.7e14` to `1.8e15` at `T = 20`.
- **The realized amplification is far smaller.** It is `12.1` and `15.8`
  (supremum `19.2`) along the exact trajectory, and `12.98` and `51.04`
  along T-bare.
- **Linearizing about the exact trajectory is inadmissible as a route.** It
  feeds an exact reference into the propagation. It also fails as an
  approximation: its error relative to the closure's departure is `0.17` at
  `T = 4` and `1.07` at `T = 20`.

---

## 4. The output contract

**Sources:** r2 sections 0, 2.5 and 3; pricing sections 0 and 2 to 5.

- **The contract.** O3, from the draft's NISQ and fault-tolerant layers:
  selected observables, here the site-0 occupation per spin and the Eq. (22)
  energy at `T = 4` and `20`, to `epsilon = 1e-2` absolute.
- **Unreachable against T-physical for any route that integrates the
  closure.** The closure's own error against the exact cutoff-16 reference,
  pointwise at `T`:

  | observable | `T = 4` | `T = 20` |
  |---|---|---|
  | site-0 occupation | `0.028` | `0.41` |
  | energy | `0.015` | `0.015` |

  Every entry exceeds `1e-2`.
- **The two admissible contracts.**
  - **(a) Against T-bare.** An integration-accuracy contract for the archive
    equations, with the closure's model error reported on its own line.
  - **(b) Against T-physical, by direct simulation.** Direct simulation of
    the exact dimer, which does not integrate the archive equations.
- **Route (a)**: the re-centred windowed order-3 lift (pricing sections 2
  to 4).
  - **Admissible readout noise.** It holds the contract (95th percentile
    over 50 draws, both observables, both horizons) only at 19 re-lifts.
    With fewer re-lifts the truncation error alone breaks it (energy
    `1.6e-2` at 5, `1.1e-2` at 10). With 40, 41 noisy readouts accumulate
    (energy `1.1e-2` at `sigma = 2e-4`).
  - **Readout requirement.** All 31 coordinates are read at every restart,
    with Gaussian error `sigma <= 2e-4` per coordinate; the energy at
    `T = 20` binds, at `9.2e-3`. That is equivalent to at least `2e8`
    Hadamard-test shots per amplitude.
  - **Ledger:**
    - order 3 gives `30,783` dimensions and `614,294` nonzeros per window;
    - `alpha_BE` is 488 (sparse access) or 32 to 41 (LCU bound);
    - first-block weight `0.77` to `0.97` at `gamma = ||x_r||`;
    - readout of `31 pi ||x_r|| / sigma = 1.1e6` to `1.4e6` calls per
      restart;
    - total about **`1.1e9`** calls (LCU access) or `1.5e10` (sparse
      access), of which `2.4e7` are readout window solves.
  - **Classical baseline, same output and target.** DOP853 on the committed
    closure reaches `t = 20` in **`0.19 s`** (1,322 right-hand-side
    evaluations), with error `1.4e-9` on the site-0 occupation at
    `T = 20`.
- **Route (b)**: direct simulation (pricing section 5).
  - **12 qubits**; `T (||H|| + max|f|)` is 102 at `T = 4` and 511 at
    `T = 20`.
  - Readout is `pi ||O|| / epsilon`: 314 calls for the site-0 occupation and
    7,754 for the energy, normalized by `||H|| = 24.68`.
  - Total about **`4.9e6`** calls.
  - Its error is only the phonon cutoff (16 against 20): at most `2.6e-4` on
    the occupation and `9.6e-5` on the energy.

---

## 5. Conclusion

**On the L = 2 Holstein dimer there is no quantum case for integrating the
archive equations, Eqs. (14a)-(14e) with the coherent-field update (14e),
under either contract.**

**Against the physical reference.**

- No closure-integrating route can meet the contract: the closure is off by
  `0.015` to `0.41` before any quantum error is added.
- The only route that meets it, direct simulation of the dimer's own
  Hamiltonian on 12 qubits, does not integrate the archive equations.

**Against the archive equations themselves (T-bare).** Of the three layers
of the draft's pedagogical solver map:

- **The fault-tolerant lifting layer** meets the contract only after it is
  turned into a windowed scheme that is re-centred at a classically read-out
  state 19 times. The origin-centred lift does not converge past `t ~ 2`.
  That scheme costs about `1.1e9` calls to reproduce a `0.19 s` classical
  integration, and its readout is itself a classical description of the
  state at every restart.
- **The fault-tolerant linearization layer** carries a recovery factor of
  `1e3` (`T = 4`) to `1e15` (`T = 20`) in Euclidean coordinates, reducible
  only partly by a constant metric. It needs a base trajectory: either the
  exact one, which is inadmissible, or the closure's own, which is what is
  being computed.
- **The NISQ residual layer was not priced in this thread.** The closure
  error excludes it under the physical contract. Under the T-bare contract
  it would face the same `0.19 s` classical baseline and the same
  per-restart readout of the 27 quantum-target coordinates.

**What a case at larger L would have to show.**

- **That fewer than all coordinates need reading at each restart.**
  - Counting the same blocks at `L` sites gives
    `2L^3 + 3L^2 + L + 1` closed coordinates: 31 at `L = 2`, 85 at
    `L = 3`, 181 at `L = 4`. This is a counting formula; no closure was
    built beyond `L = 2`.
  - That count is polynomial, and so is the classical cost of integrating
    the bare closure.
  - Reading all coordinates at every restart is that classical cost, times
    the amplitude-estimation overhead `pi ||x|| / sigma`.
  - A case would need re-centring, which the lift requires (section 3.2),
    without a full classical readout. Nothing here supplies one.
- **That a bounded restart count still holds the contract.** At `L = 2`
  only 19 re-lifts work at order 3: 10 fail on truncation and 40 on
  accumulated readout noise.
- **That it beats direct simulation of the physics.** Direct simulation
  needs about `2 log2 L + 4.1 L` qubits (dense: one up and one down
  electron, a 17-level mode per site) and carries no closure error. The
  exact state dimension grows as `L^2 17^L`: `1,156` at `L = 2` and
  `44,217` at `L = 3`. That is what makes classical exact propagation costly
  (`3.06 s` and about `137 s` to `t = 20`, 2026-09-20 scaling record), not
  the quantum register.
- **For the repaired closure, that it accelerates the repair, not the
  integration.**
  - The repair's semidefinite step is `0.80` of the classical wall (draft
    block C).
  - Its dictionary rows grow quartic to quintic in `L`: 36, a counted 281
    and 1,192 at `L = 2, 3, 4`. A locality-restricted dictionary has linear
    rows, but its transitive hull under the Holstein commutator grows at
    exponents `5.26`, `5.05`, `5.02` (2026-09-20 scaling record).
  - The 18 GB memory contract caps the conic block at 87 extension rows. The
    declared dictionary exceeds it already at `L = 3` (281 rows). The
    locality-restricted dictionary (71 and 106 rows at `L = 3, 4`) exceeds
    it between `L = 3` and `4`, and that dictionary does not close.
  - No route in this thread addresses that step.

---

## 6. Every claim withdrawn during verification, with its replacement

**Sources:** feasibility Corrections (`475c47a7`); r2 section 1
(corrections 1 to 28).

| # | withdrawn | replacement |
|---|---|---|
| 1 | Task 1: the lifting layer is "not admissible" | Not admissible under the Liu et al. sufficient condition and the tested origin-centred truncations only; a failed sufficient condition is not a no-go |
| 2 | "The closure diverges from exact data at `t = 433.8964`" | That is the `rtol 1e-10` value from the exact `t = 60` state; converged it is `434.19`. From the exact correlated preparation at `t = 0`, the escape is at `141.4059` |
| 3 | "Pointwise tail ratio, median 0.72" (as the Task 3 quantity) | `0.72` is on exact states. Along the closure it is `0.547` (`t <= 4`) and `0.474` (`t <= 20`), the same at every order |
| 4 | The manuscript target "`F_0 + (A_1 + f D) x + A_2 (x (x) x)`" | Specific to the affine closed chart; in the draft's vectorization `F_0 = 0`, and only the cubic term is wrong |
| 5 | "The linear part preserves a 39-dimensional family" | 41 (the 39 was a column-scaling artifact) |
| 6 | "The only preserved tensors are rank 2 on the decoupled centre mode", "no Darboux or Lie-Poisson map", "all the structure there is" | True only for constant and affine tensors in the closed chart (ranges in `(Re B_+, Im B_+, N_++)`). At quadratic order `F^w/(2 omega)` is Poisson and generates `F` with the Eq. (22) energy; `F^w + u^v` (rank 4) is preserved |
| 7 | "The only exact reductions" | Only the linear ones; nonlinear level-set reductions (functional rank 5) are not assessed |
| 8 | "A polynomial of odd degree is unbounded" (why no first integral bounds the off-diagonal `C`) | All nine first integrals are invariant under translations in a 6-dimensional subspace of the 8 off-diagonal `C` coordinates |
| 9 | The commutator non-closure (a hand derivation) | Verified numerically: 92 of 595 pairs leave the span |
| 10 | "Trace of the Jacobian is zero" | Stronger: every diagonal entry `dF_i/dx_i` is identically zero |
| 11 | KvN "434 to 682 qubits, `T ||L||` 5e5 to 6e9" as a cost | A heuristic; box-norm `T ||L|| >= 2.0e7`, `3.9e8`, `7.3e10`; compression-resolved 558, 682, 1,023 qubits; envelope 651, 1,984, 5,518 |
| 12 | Baseline "12 or 19 qubits, `T ||H||` 510 to 680" | The dimer: 12 qubits, `102`, `511`, `1,533` at matched horizons. The 19-qubit, 680 figure is the three-site chain |
| 13 | "Realized amplification 12 to 51" | Mixed trajectories: exact `12.1`, `15.8` (supremum `19.2`); T-bare `12.98`, `51.04` |
| 14 | "Overhead at least `e^p/Lambda`, about `1e2` and `1e14`" | Forced-flow recovery factor `1.2e3` to `1.5e3` (`T = 4`) and `8.7e14` to `1.8e15` (`T = 20`) |
| 15 | "A time-dependent metric would be needed" | A constant metric gives `2.09` and `11.8` (theorem form along T-bare) |
| 16 | "The metric in which `A_1` is skew-adjoint", "7.23", "at most 7.2" | Not unique (a 46-dimensional family); the driven supremum is at least `7.27` in the task-1 metric and `6.21` in another member |
| 17 | "No common metric" from the SCS value `1.9e-8` | Proved by a nilpotent element of the 91-dimensional Lie algebra (`||exp(1000 X)|| = 647`) |
| 18 | Windowed re-lift as "a lower bound on any windowed scheme", "80 restarts" | Holds only for origin-centred lifts (re-centred: `4.1e-8`, `6.3e-7`); 79 re-lifts plus the final readout |
| 19 | "The propagated tail is measured exactly", "nothing else" | Duhamel's identity, evaluated to quadrature accuracy (Simpson `1.1e-9` to `4.1e-9`); it validates the pipeline |
| 20 | "Already of order one" | `2.0` to `2.2` (`T = 4`), `19.6` to `127` (`T = 20`) |
| 21 | First-block weights with no encoding | Kronecker, constant block excluded, `gamma = 1`; `0.525`, `0.249`, `0.080` at `gamma = ||x_0||`; monomial encoding `0.347`, `0.111` |
| 22 | "Independent verifications" | They were self-consistency checks; replaced by the Hamiltonian parity and second transcription, which catch the mutation |
| 23 | "The draft supports four contracts" | Three anchored (trajectory diagnostic, selected observables, invariants and residuals); the terminal-state contract was an addition; spectra, a representation claim, the characterization set and `M_ij`, `V_i` were missed |
| 24 | "O3 is the only contract whose readout does not scale with `31 N_t`" | O2, O3 and O4 all have readout independent of `N_t` |
| 25 | The recommendation "O3 at `1e-2` against the physical reference" | Unreachable for any closure-integrating route; restated as the choice between contracts (a) and (b) |
| 26 | "Phonon occupation `N00 + N11`" | That is the fluctuation part; the total, `|B|^2 + Tr N`, is quadratic (`|B|^2` is 79 to 88 percent) |
| 27 | "Block 2 of a lift supplies `x (x) x` directly" | False in a truncated lift (Frobenius discrepancy `11.3` to `802`) |
| 28 | "No linear map from `X_K` evaluates the repair's cones" | Withdrawn for the retained joint Gram, whose entries are exact degree-2 polynomials of the 31 coordinates |
| 29 | `kappa_stack` "1.1e4 to 1.16e7" | Mixed targets; for T-bare `12,071`, `166,189`, `11,558,518` |
| 30 | Linearization about the exact trajectory, `0.17` and `1.07`, as a route | Inadmissible (an exact reference as input); a diagnostic only |
| 31 | "Certified `Lambda`" from the terminal adjoint (`39.30` at `T = 20`) | The tube needs the pair supremum: `51.04`, or `54.10` inflated |
| 32 | "Nonresonance fails automatically at order three" | It fails at order two with nonzero coupling (33 of 79) |
| 33 | Reduced-system status quoted with full-system ranks | Reduced: 24 / 24 / 28; all 4 zero modes receive the quadratic term |
| 34 | Status "uncommitted"; two hash suffixes | Committed at `a23ff642`; hashes corrected in r2 |

---

## 7. What remains unmeasured

**Sources:** pricing section 6; r2 section 8 and "Unresolved".

- **Circuits.** There are no circuits: none for state preparation (of `x_r`,
  the lift or the forcing), none for block encoding of the lifted generators,
  and no gate counts. The LCU normalization `alpha_H` of the boson-encoded
  dimer Hamiltonian for route (b) is not computed; only `||H|| = 24.68` is
  used.
- **Solver constants.** `C_max = sup ||exp(G t)||` per window, the
  polylogarithmic factors and the solver's discretization error are
  unmeasured. There is no certified smallest singular value for any stacked
  system.
- **The readout model.**
  - The noise is unbiased and independent per coordinate.
  - Sign recovery is omitted for coordinates near zero: the encoded part
    `x_0/2` of the occupation changes sign 7 times to `t = 20`.
  - The norm `||x||` is assumed known.
  - There is no union bound over the 620 estimates of route (a).
  - 50 draws per row: the 95th percentile at the admissible boundary
    (`9.2e-3` against `1e-2`) carries about 10 percent sampling uncertainty.
- **What the certificates assume.** The pair supremum `Lambda` takes the
  base trajectory as exact, uses a sampled `max ||F||` and a Richardson RK4
  allowance. The order-4 adjoint at `T = 20` was not run.
- **Unassessed structure.** Nonlinear level-set reductions, and
  Lie-Poisson structures in non-affine charts (such as raw moments).
- **KvN.** Boundary flux out of the box, a sufficient grid size, and the
  cost of preparing the packet.
- **Configuration.** One preparation (exact correlated), one parameter set,
  horizons 4 and 20. Nothing was run past the escape at `t = 141.41`, and no
  closure beyond `L = 2`.
