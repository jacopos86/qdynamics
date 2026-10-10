# Can the archive equations of motion be integrated on a quantum computer? (2026-10-10)

Branch `claude/paper-v-routes-20260914`, worktree
`Holstein_test.worktrees/claude-paper-v-routes-20260914`, HEAD `b9f99271`.
Exploratory local work, not promoted evidence; no default changed; nothing
committed; the manuscript `MATH/paper_details/paper_V_high_u_gkba.tex` is not
edited. Exact references are reporting inputs only.

Object of study: the committed archive right-hand side `closed_scalar_rhs`
(`paper_5/src/paper5/stability/matrix_reference.py`), Eqs. (14a)--(14e) of
Riva, Simoni and Ping in the 31 closed scalar coordinates
`CLOSED_SCALAR_STATE_NAMES`, blocks `rho` (3), `B` (4), `N` (4), `A` (6), `C`
(14), at the declared configuration `DimerParameters()`: `t_hop = 1`,
`gamma = 0.5` (`omega_ph = 0.5`), `lambda_ep = 1.5` (`g = 0.61237`), drive
amplitude `1`, pulse width `1`, drive `f(t) = 2 sin(pi t/4) exp(-t^2/2)`.

## Verdict on Task 1

**The fault-tolerant lifting layer is not admissible for these equations as
they stand, and the failing property is dissipativity, by a structural
margin.** The Carleman sufficient condition of Liu et al. needs every
eigenvalue of the linear part to have negative real part, and `R < 1` needs
`Re lambda_1 <= -4.95` at the exact correlated preparation (`-4.10` at the
Hartree--Fock zero-correlation preparation). The measured value is
`Re lambda_1 = +1.05e-16`, zero to rounding. The failure cannot be removed by
choosing another expansion point or a linear change of coordinates. The
archive vector field is exactly divergence-free: `tr J(t, x) = 0` for every
state and time, so the real parts of the Jacobian eigenvalues always sum to
zero. Without that convergence guarantee, the order-`K` Carleman truncation
measured from the exact correlated preparation converges in `K` only over
`t <= 2`. Its error is about `1` in coordinate units at `t = 4` for every
`K = 1..4`, and grows with `K` after that: the maximum absolute coordinate
error against the closure's own trajectory over `t = 0` to `20` is `12.3`,
`72.1` and `387` at `K = 2, 3, 4`. Both decisive horizons of the draft,
`t = 4` and `t = 20`, lie at or beyond that window.

Two properties pass and are more favourable than the draft assumed. The
archive right-hand side has **no cubic part**: it is exactly quadratic, which
is the form Liu et al. treat. The quadratic tensor is sparse, with 69 nonzero
monomial coefficients, all in the `rho` and `C` equations.

## (a) The explicit polynomial form

### Extraction and verification

`paper_5/src/paper5/stability/quantum_target_form.py`,
`extract_closed_polynomial_form`. The method is exact polynomial
interpolation of `closed_scalar_rhs` at the integer points `0`, `s e_i`
(`s = -1, 1, 2`), `+-e_i +- e_j` and `e_i + e_j + e_k`. That is
`1 + 3*31 + 3*465 + 4495 = 5,984` evaluations, one per monomial of degree at
most three. Every coefficient is a fixed integer combination of these
evaluations.

- **The drive-free part** is extracted at `drive_amplitude = 0`.
- **The drive part** is extracted at `t = 0.7` (`f = 0.81792`) and at
  `t = 1.9` (`f = 0.32793`), as the full polynomial at that time minus the
  drive-free polynomial.
  - Its constant and quadratic parts are `0.0` exactly, and its cubic part is
    at most `1.1e-16`.
  - Its linear part divided by `f(t)` agrees between the two times to
    `2.2e-16`.
  - **So time enters only through `f(t) D x`.** The form is

    `dx/dt = F_0 + (A_1 + f(t) D) x + A_2 (x (x) x)`, with `A_3 = 0`.

- **Reconstruction.** The extracted form was compared with `closed_scalar_rhs`
  on 500 fresh random states per scale. Coordinates are independent Gaussians
  with standard deviation `0.1`, `1` and `3`, and times are uniform on
  `[0, 8]`. The maximum absolute error is `2.8e-16`, `1.8e-14` and `3.6e-13`.
  Relative to the largest velocity component, it is `4.4e-16`, `1.4e-15` and
  `1.7e-14`. The drive-free part alone gives the same figures. The drive part
  alone has maximum absolute error `2.2e-16`, `3.8e-15` and `2.8e-14`.
- **Rounding gap.** Across all tensors, the largest coefficient at or below
  `1e-10` is `2.2e-16`, and the smallest above it is `0.15309`. The nonzero
  pattern is therefore unambiguous.

**Why there is no cubic part (measured, then read off the code).** In
Eq. (14d), the normal particle source is `-i N[q,q'] (1 - rho) g^T rho` and
the normal hole source is `+i N[q,q'] rho g^T (1 - rho)`. They sum to
`-i N[q,q'] [g^T, rho]`, and the `rho g rho` products cancel. The only
remaining product of two `rho` factors is the bare Pauli source
`-i (1 - rho) g rho`, which is degree two. The cubic term `C(x,x,x)` in the
draft's Eq. (`eq:paper_v_force_polynomial`) and the `A_3` of the lifting
layer are therefore identically zero for these equations. The manuscript is
not edited.

### Degree per block and which blocks couple

Output block on the left. The count is the number of nonzero monomial
coefficients for each input block or input-block pair.

| output | degree | constant | linear (`A_1`) | quadratic (`A_2`) | drive (`D`) |
|---|---|---|---|---|---|
| `rho` (3) | 2 | -- | `rho` 2, `C` 8 | `(rho)(B)` 4 | `rho` 2 |
| `B` (4) | 1 | 2 | `B` 4, `rho` 2 | -- | -- |
| `N` (4) | 1 | -- | `C` 9 | -- | -- |
| `A` (6) | 1 | -- | `A` 6, `C` 14 | -- | -- |
| `C` (14) | 2 | 3 | `C` 30, `rho` 8 | `(B)(C)` 16, `(rho)(N)` 24, `(rho)(rho)` 17, `(rho)(A)` 8 | `C` 8 |

- **`F_0`** has five nonzero entries:
  - `-g = -0.61237` on `coherent_0_imag` and on `coherent_1_imag`;
  - `-g/4 = -0.15309` on `correlation_shared_trace_imag` and on
    `correlation_0_diag_difference_imag`;
  - `+g/4` on `correlation_1_diag_difference_imag`.
- **`D`** is skew-symmetric. Its 10 entries are `+-1` rotations of the
  electronic coherence (real, imag) and of the off-diagonal entries of both
  correlation matrices.

Nonzero counts and norms. Spectral norm means the largest singular value of
the `31 x 31^p` matricization acting on `x^{(x)p}`, with the symmetric
representation of `A_2`; for `F_0` it is the Euclidean norm.

| tensor | nonzero entries | nonzero monomials | spectral norm | Frobenius norm |
|---|---|---|---|---|
| `F_0` | 5 | 5 | 0.90571 | 0.90571 |
| `A_1` (drive-free) | 83 | 83 | 4.0473 | 9.8742 |
| `D` (drive) | 10 | 10 | 1.0000 | 3.1623 |
| `A_2` | 129 | 69 | 1.9954 | 5.3099 |
| `A_3` | 0 | 0 | 5.0e-16 | 8.4e-16 |

Block norms (`task1_form.json`, `block_norms`):

- `A_1`, spectral norm of each sub-block:
  - `rho<-rho 4.0`, `rho<-C 1.2247`;
  - `B<-rho 0.8660`, `B<-B 0.5`;
  - `N<-C 2.2184`;
  - `A<-A 1.0`, `A<-C 2.2184`;
  - `C<-rho 0.6124`, `C<-C 2.9873`.
- `A_2`, Frobenius norm of each sub-tensor:
  - `rho<-(rho)(B) 1.7321`;
  - `C<-(rho)(rho) 1.6417`, `C<-(rho)(N) 2.1213`, `C<-(rho)(A) 2.4495`,
    `C<-(B)(C) 3.4641`.

**Under the draft's working split** (Eqs. (14a)--(14d) as the quantum
target and Eq. (14e) as a classical update of `B`):

- The `(rho)(B)` and `(B)(C)` terms become linear terms with coefficients
  `B(t)`.
- The quadratic part of the 27 quantum-target coordinates is the 49
  monomials `(rho)(rho)`, `(rho)(N)` and `(rho)(A)`, all in the `C` equation.

## (b) The linear part: spectrum, character, and the Liu et al. ratio

### Spectrum of the drive-free linear part `A_1`

- **All 31 eigenvalues have `|Re lambda| <= 2.2e-16`.**
  - Their imaginary parts are `0` (5-fold), `+-0.45685`, `+-0.5` (4-fold
    each), `+-1` (3-fold each), `+-1.5`, `+-1.60939`, `+-2.18890`, `+-2.5`
    and `+-2.58067`.
  - The spectral radius is `2.58067`.
- **`A_1` is semisimple.**
  - Every eigenvalue's algebraic multiplicity equals its geometric
    multiplicity.
  - The eigenvector condition number is `55.4`.
  - `A_1` is skew-adjoint in the inner product `G = Re(V^{-H} V^{-1})`:
    `||A_1^T G + G A_1|| / (||G|| ||A_1||) = 1.0e-16`, with `cond(G) = 3,066`.
- **In the Euclidean metric of the closed coordinates `A_1` is not skew.**
  - Its symmetric part has eigenvalues from `-1.5978` to `+1.5978`: nine
    negative, thirteen zero and nine positive.
  - The Euclidean logarithmic norm is therefore `+1.598`. A Euclidean norm can
    grow transiently at up to that rate under `A_1` alone.

**Character: anti-Hermitian-like (conservative), not dissipative and not
mixed.** The drive does not change this:

- `D` is Euclidean skew-symmetric with eigenvalues `+-i` and `0`.
- Over the drive's whole range, `f in [-4.68e-5, 0.8664]` (maximum at
  `t = 0.908`), `A_1 + f D` keeps `|Re lambda| <= 7.5e-16`.

### The structural reason no expansion point can be dissipative

The trace of the Jacobian vanishes identically:

- `tr A_1 = 0.0`, `tr D = 0.0`;
- `d(tr J)/dx = 2 sum_i A_2[i, i, :] = 0.0` exactly.

So `tr J(t, x) = 0` at every state and time. The archive field is
divergence-free in the closed coordinates. The sum of the real parts of the
Jacobian eigenvalues is zero at every expansion point and is invariant under
any invertible linear change of coordinates. Hence `Re lambda_1 >= 0`
everywhere, and the premise of the Liu et al. theorem fails wherever the lift
is centred.

Measured Jacobian spectra:

| point | `Re lambda` of the 31-coordinate Jacobian | 27 quantum-target coordinates, `B` frozen |
|---|---|---|
| exact correlated preparation | `-0.15713` to `+0.15713` (eigenvector condition `23.1`) | `|Re lambda| <= 9.4e-16` |
| Hartree--Fock zero-correlation preparation | `|Re lambda| <= 6.5e-16` | `|Re lambda| <= 9.4e-16` |

The pair at the exact preparation is therefore carried by the `B` feedback.

### The ratio `R`

**Definition used.** Liu, Kolden, Krovi, Loureiro, Trivisa and Childs, PNAS
118, e2026805118 (2021), arXiv:2011.03185, Eq. (2.2), checked against the
arXiv text:

```text
R = (||u_in|| ||F_2|| + ||F_0|| / ||u_in||) / |Re lambda_1|
```

- `||.||` is the spectral norm and `F_0 = max_t ||F_0(t)||`.
- The theorem assumes `F_1` and `F_2` time-independent, `F_1` diagonalizable,
  and `Re lambda_n <= ... <= Re lambda_1 < 0`.
- The paper proves efficiency for `R < 1` and gives a worst-case
  intractability lower bound for `R >= sqrt(2)` over general quadratic ODEs.
- The form stated in the task, `||A_2|| ||x_0|| / |Re lambda_1|`, is the same
  quantity without the `F_0` term. Here `F_0 != 0`, so both are reported.
- The task names the authors as Liu, Kolden, Krovi, Lloyd, Montgomery and
  Childs. The paper's authors are Liu, Kolden, Krovi, **Loureiro, Trivisa**
  and Childs.

Applied to `F_1 = A_1`, `F_2 = A_2` (`1.9954`), `F_0` (`0.90571`) and the two
preparations in the 31 closed coordinates:

| preparation | `||x_0||` | `||A_2|| ||x_0||` | `||F_0|| / ||x_0||` | numerator | needed for `R < 1` | measured `Re lambda_1` |
|---|---|---|---|---|---|---|
| exact correlated (cutoff 16 ground state) | 2.28307 | 4.5557 | 0.3967 | 4.9524 | `Re lambda_1 < -4.952` (`-4.556` without `F_0`) | `+1.05e-16` |
| Hartree--Fock zero correlation | 1.80278 | 3.5973 | 0.5024 | 4.0997 | `Re lambda_1 < -4.100` (`-3.597` without `F_0`) | `+1.05e-16` |

**Plainly:** the archive equations do not satisfy the condition.

- `Re lambda_1` is zero to rounding, with the wrong sign for the premise.
- `R` is formally `4.7e16`. That number is not meaningful, because the
  hypothesis it rests on fails.
- The missing quantity is a damping rate of at least `4.95` per time unit at
  the exact correlated preparation (`4.10` at Hartree--Fock).
- The equations also break a second, lesser hypothesis: `F_1` is
  time-dependent through the drive (`A_1 + f(t) D`). This matters only while
  `f` is on; `|f(8)| = 6e-30`.

## (c) The Carleman truncation at `K = 2, 3, 4`

### Lifted dimension

The lift is `X_K = (x, x^{(x)2}, ..., x^{(x)K})`; the constant block is
excluded. Because `A_3 = 0`, the lifted matrix is block-tridiagonal: `F_0`
couples `k <- k-1`, `A_1 + f(t) D` couples `k <- k`, and `A_2` couples
`k <- k+1`.

| `K` | Kronecker-power dimension (31) | distinct monomials (31) | Kronecker (27, working split) | monomials (27) |
|---|---|---|---|---|
| 1 | 31 | 31 | 27 | 27 |
| 2 | 992 | 527 | 756 | 405 |
| 3 | 30,783 | 5,983 | 20,439 | 4,059 |
| 4 | 954,304 | 52,359 | 551,880 | 31,464 |
| 5 | 29,583,455 | 376,991 | 14,900,787 | 201,375 |

### The residual the truncation leaves, pointwise on the stored exact-contracted trajectory

The stored trajectory is
`output/local_runs/paper_v_projective_dictionary_energy_20260915e/exact_t20.npz`:
cutoff-16 exact driven reference, 8,001 samples at spacing `0.0025` to
`t = 20`. Regenerated in process, it agrees with the cache to `3.6e-15`, and
its `t = 0` row equals `exact_ground_closed_scalar_coordinates` exactly.

At a state `x`, the lifted velocity of block `k` under the closure's own field
is `sum_m x^{(m-1)} (x) F(t,x) (x) x^{(k-m)}`. The order-`K` truncation keeps
it exactly in blocks `1..K-1`. In block `K` it drops `A_2 (x (x) x)` inserted
at each of the `K` positions. The table reports the norm of what is dropped
from block `K`, divided by the norm of block `K`'s full lifted velocity. It
uses the closed form `k ||v||^2 ||x||^{2(k-1)} + k(k-1) <x,v>^2 ||x||^{2(k-2)}`,
checked against explicit tensors in the tests.

| `K` | on exact states, `t <= 4`: median / max | on exact states, `t <= 20`: median / max | on the closure's own trajectory, `t <= 20`: median / max |
|---|---|---|---|
| 2 | 0.718 / 1.119 | 1.157 / 1.920 | 0.476 / 1.119 |
| 3 | 0.718 / 1.119 | 1.155 / 1.914 | 0.474 / 1.119 |
| 4 | 0.717 / 1.119 | 1.153 / 1.908 | 0.473 / 1.119 |

**The dropped part is as large as what is kept, and it does not shrink with
`K`.** The reason is that the quadratic term is a large share of the closure
velocity at physical states. On the exact states over `t <= 4`,
`||A_2 x (x) x||` has median `0.489`, against `0.684` for the full closure
velocity. A Carleman truncation can therefore converge only through the
`K - 1` couplings the dropped term needs to reach block 1. Without
dissipation that buys a finite time window, which the solution error below
measures.

For scale, the closure's own defect on the same exact states is smaller than
that dropped term on average, but of the same order:

- The defect is the exact contracted velocity minus the closure velocity at
  the exact state.
- Its norm has median `0.427` over `t <= 4` and `0.351` over `t <= 20`, and
  maximum `0.508` and `0.527`.
- The exact velocity's norm has median `0.507` and `0.442` over the same
  windows.
- At `t = 0` the defect is `0.4097`, the closure velocity at the exact ground
  state.

The full residual of the lifted exact trajectory in the truncated system
(defect plus truncation), relative to the closure-field lifted velocity, has
median `1.14` to `1.17` over `t <= 4` and `1.34` to `1.38` over `t <= 20`, for
`K = 2..4`.

### The truncation error of the solution

Setup:

- **Truncated system.** The order-`K` truncated linear system starts from the
  lift of the exact correlated preparation and is integrated by RK4 at
  `h = 0.0025` on the stored grid.
- **Comparison.** Its first block is compared with the closure's own
  trajectory from the same state. That trajectory comes from DOP853 at
  `rtol 1e-12`, `atol 1e-13`, `max_step 0.01` on the extracted form.
- **Check on that trajectory.**
  - It matches the same integration on the committed `closed_scalar_rhs` to
    `7.8e-16` over `t <= 4`.
  - Its all-coordinate scalar RMS against the exact reference reproduces the
    record's plain-archive row, `1.10890e-1`, `1.38174e-1` and `1.93196e-1`
    at `t = 4, 8, 20`, to every printed digit.
- **Integration error.**
  - Step halving changes the first block by at most `1.8e-8` (`K <= 3`, to
    `t = 20`).
  - At `K = 4`, `h = 0.005` against `h = 0.0025` differs by `2.8e-9`
    (to `t = 4`).
  - Every truncation error in the table below is more than `4e5` times the
    integration error at the same order.

Maximum absolute coordinate error of the first block against the closure's
own trajectory, over `t = 0` to the horizon:

| `K` | `t <= 1` | `t <= 2` | `t <= 4` | `t <= 8` | `t <= 20` | first time above `1e-2` | above `1e-1` | above `1` | wall |
|---|---|---|---|---|---|---|---|---|---|
| 1 (linear part only) | 0.2612 | 0.6444 | 0.9571 | 1.540 | 1.663 | 0.0475 | 0.500 | 4.063 | 0.2 s |
| 2 | 0.05127 | 0.3014 | 0.9808 | 2.866 | 12.31 | 0.610 | 1.248 | 4.028 | 0.8 s |
| 3 | 0.007574 | 0.1194 | 1.036 | 5.377 | 72.13 | 1.068 | 1.923 | 3.503 | 7.7 s |
| 4 | 0.001203 | 0.06402 | 1.001 | 13.07 | 387.1 | 1.383 | 2.170 | 3.938 | 275.7 s, 214 MB |

How the error changes with each added order (`K+1` against `K`):

- `0.20`, `0.15` and `0.16` at `t <= 1`;
- `0.47`, `0.40` and `0.54` at `t <= 2`;
- `1.02`, `1.06` and `0.97` at `t <= 4`;
- `1.9`, `1.9` and `2.4` at `t <= 8`.

The window in which the truncation converges therefore closes near `t = 4`.

The all-coordinate scalar RMS against the exact reference is as follows:

- At `t = 4`: `0.341`, `0.189`, `0.204` and `0.146` for `K = 1..4`, against
  the closure's own `0.1109`.
- At `t = 20`: `0.554`, `2.33`, `8.05` and `48.8`, against the closure's own
  `0.1932`.

No integration reached the `1e4` magnitude event by `t = 20`.

## What a reader should not conclude

- **This does not show that every quantum route for these equations fails.**
  It shows that Carleman linearization with the Liu et al. guarantee does not
  apply, and that the plain truncation converges only over a window of about
  `t = 2`. A windowed scheme, re-lifted every unit or so, would need the state
  read out at each restart. That is the measurement burden the lifting layer
  was meant to avoid. It is not costed here.
- **The dissipativity failure is not specific to the expansion point or to
  the coordinates.** The trace argument covers every expansion point and every
  invertible linear change of coordinates. The truncation error of the
  solution is also unchanged by any invertible linear change of coordinates,
  because the lift transforms by a block-diagonal similarity. It does depend
  on the expansion point, which was not varied here: every lift above is
  centred at the origin of the closed coordinates. Nonlinear changes of
  variables and added artificial damping are not covered, and both would
  alter the equations being integrated.
- **The draft's working split does not rescue the premise.** The
  27-coordinate block with `B` held at its prepared value also has trace zero
  and a purely imaginary spectrum at both preparations.
- **The pointwise residual table is not an error bound.** It shows that the
  dropped term is of order one; the solution table is the error.

## Reproduction

```bash
cd /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914
PY=/Users/lordskyler/Holstein_test/.venv/bin/python
DIR=output/local_runs/paper_v_quantum_target_form_20261010
PYTHONPATH=paper_5/src $PY $DIR/run_task1.py                    # (a), (b), pointwise (c); 46 s, 765 MB peak
PYTHONPATH=paper_5/src $PY $DIR/run_carleman.py --orders 1 2 3  # 9 s
PYTHONPATH=paper_5/src $PY $DIR/run_carleman.py --orders 4      # 276 s, 214 MB peak
PYTHONPATH=paper_5/src $PY $DIR/run_carleman.py --orders 1 2 3 --steps-per-sample 2 --tag _h000125
PYTHONPATH=paper_5/src $PY $DIR/run_carleman.py --orders 4 --final-time 4 --sample-stride 2 --tag _h0005_t4
$PY $DIR/check_steps.py
PYTHONPATH=paper_5/src $PY -m pytest paper_5/tests/test_quantum_target_form.py -q   # 12 passed, 6.4 s
```

Artifacts in `$DIR`:

- `task1_form.json`: every number in sections (a) and (b), and the pointwise
  part of (c);
- `task1_form.npz`: the tensors `F_0`, `A_1`, `D`, `A_2` and `A_3`;
- `task1_trajectories.npz`: exact states and exact derivatives, and the
  closure trajectory;
- `carleman_orders_*.json` and `carleman_K*.npz`: the truncation solutions;
- `check_steps.json`: the integration-step checks.
