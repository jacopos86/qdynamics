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
they stand under the Liu et al. sufficient condition and the tested
truncations (`K = 1..4`, lifted at the origin of the closed coordinates), and
the failing property is dissipativity, by a structural margin.** This scope
was narrowed on 2026-10-10 after the B-Q2 theory dossier: a failed sufficient
condition is not a proof that no quantum algorithm exists. The Carleman
sufficient condition of Liu et al. needs every
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

---

# 2026-10-10, Task 2: is the closure Hamiltonian, and what do Koopman--von Neumann and Schrödingerization cost?

Same branch and worktree; HEAD `6b27596e` at the time of writing (the
coordinator committed the task-1 files in `d110f2dd`). Exploratory, not
promoted; nothing committed by this agent; the manuscript is not edited.
Scripts and outputs are under
`output/local_runs/paper_v_quantum_target_form_20261010/` (`run_task2_*.py`,
`task2_*.json`). The closure-instability report
`agent_guidance/paper-v/overnight/closure-instability-20261010.md` (commit
`40771435`, SHA-256 `751cab34...a5ae71`) is the authority for amplification
and conditioning; this section uses its numbers.

## Answer first

1. **The drive-free archive field is not Hamiltonian with respect to the
   Eq. (22) energy for any constant or affine `J(x)`.** More generally, it is
   not Hamiltonian with respect to any constant or affine Poisson tensor in the
   closed coordinates, whatever the Hamiltonian.
   - The only affine bivectors the flow preserves have rank 2. They are
     confined to the decoupled centre-mode phonon sector.
   - Exactly true: energy conservation (as a polynomial identity),
     volume preservation, a semisimple linear part that is skew-adjoint in a
     metric, and nine independent polynomial first integrals of degree at
     most three.
   - None of those first integrals bounds the off-diagonal electron--phonon
     correlation entries, which are the coordinates that escape.
2. **The Koopman--von Neumann embedding does not need Hamiltonian structure.**
   The half-density Koopman generator is formally Hermitian for any smooth
   vector field (Joseph 2020; the B-Q2 dossier). Here the divergence is zero,
   so the generator is simply `-i F . grad`.
   - Applied to a single trajectory of the 31-coordinate closure, it needs
     about 430 to 680 logical qubits and a generator-norm-times-time of
     `5e5` to `6e9` at coordinate precision `1e-2`, depending on the horizon.
   - The problem the closure approximates is itself a Hamiltonian simulation
     on 12 qubits (two sites) or 19 (three sites), with `T ||H||` of about
     `510` and `680` at `T = 20`, and carries no closure error.
   - The closure route is dominated on every axis.
3. **Schrödingerization of the trajectory-linearized flow is not priced by
   the realized amplification.**
   - The realized amplification is 12 to 51 to `T = 20`.
   - The published recovery overhead `e^{p}` uses the pointwise logarithmic
     norm: `e^{7.3}` at `T = 4` and `e^{35}` at `T = 20` in the Euclidean
     closed coordinates.
   - The metric in which `A_1` is skew-adjoint makes this worse
     (`e^{51}`, `e^{251}`).
   - The linearization itself represents the closure's departure from exact
     to 17 percent at `T = 4` and not at all by `T = 20`.
4. **The windowed re-lift is bounded even with free, exact readout.** It
   reaches `2.6e-3` at `t <= 4` and `0.30` at `t <= 20`, at `K = 3` with 80
   restarts. Table at the end.

**Decision for the user (manuscript correction).** The draft's cubic term is
identically zero for these equations:

- the `C(x,x,x)` term in Eq. (`eq:paper_v_force_polynomial`);
- the `A_3 x^{(x)3}` term of the fault-tolerant lifting layer.

The extracted right-hand side is exactly quadratic: `A_3` is at most
`1.1e-16`, and the Eq. (14d) normal particle and hole sources combine to
`-i N [g^T, rho]`.

-> **Recommend** stating the target as
`F_0 + (A_1 + f(t) D) x + A_2 (x (x) x)` and dropping the cubic layer in
both places. The manuscript is untouched until you decide.

## (a) The Hamiltonian and Poisson tests

**The energy.** The Eq. (22) energy `H` was interpolated exactly from
`matrix_total_energy` and reconstructs it to `4.3e-14` on 300 random states.

- It is quadratic in the closed coordinates.
- Its gradient depends only on `electron_delta_n` and the four `B`
  coordinates; the Hessian has rank 5.
- It is linear in `phonon_00` and `phonon_11` and in the real parts of the
  correlation shared trace and diagonal differences.
- It contains no off-diagonal correlation entry and no anomalous coordinate.

| test | result | precision |
|---|---|---|
| `grad H . F == 0` as a polynomial identity | **holds**, as an identity on all of R^31 | largest coefficient of the cubic polynomial `grad H . F` is `7.4e-17`; relative `1.2e-16` on 2,000 random states (s.d. 0.1, 1, 3) and `1.1e-16` on 801 exact states |
| the drive's power | `grad H . (D x) = -4 electron_coherence_imag` exactly | -- |
| divergence | `tr J(t, x) = 0` identically (task 1; the instability report finds at most `6.1e-13`) | exact |
| antisymmetric `J(x)` with `J grad H = F` exists | **trivially yes**: `J_min = (F grad H^T - grad H F^T)/|grad H|^2`. A Poisson tensor also always exists locally near a point with `F != 0` (flow-box theorem), so only a restricted class of `J` can carry content. | -- |
| Jacobi identity for `J_min` | **fails** at all 101 sampled exact states: the component of `[F, grad H/|grad H|^2]` outside the plane of the two fields is `0.83` to `1.00` of that bracket (median `0.97`) | -- |
| linear first integrals | exactly one: `ell = Re(correlation_shared_trace) + (g/6)(N00 + N11 + 2 Re N01)`. It is also drive-invariant. | singular value `1.8e-15`, next `0.458` |
| constant `J`, any Hamiltonian: test `J_F(x) J + J J_F(x)^T = 0` for all `x` | the linear part alone preserves a 39-dimensional family, including the coordinator's `J_0 = A_1 G^{-1}` (antisymmetric to `1.2e-14`, rank 26, so `A_1 = J_0 G` is a Hamiltonian matrix). The full field preserves **one** tensor, of **rank 2**: the canonical pair of the decoupled centre-mode coherent field `B_0 + B_1`. The quadratic part breaks `J_0`'s invariance at relative `0.31`. Since `F(x)` spans 30 dimensions, **no constant Poisson tensor generates `F`.** | singular value `7.9e-16`, next `0.078` |
| affine `J(x) = J_0 + sum_k x_k J_k` (Lie-Poisson on any 31-dimensional Lie-algebra dual, with or without a constant cocycle), any Hamiltonian: test `L_F J = 0` | over 14,880 unknowns and 245,520 coefficient equations the nullspace has dimension **4**. Every element has rank 2 and is supported on the `B` and `N` coordinates. `F` lies outside every element's range (relative residual at least `0.34`), and a random combination has rank 2. **No affine Poisson tensor generates `F`.** | relative eigenvalues of `M^T M` at most `4.7e-16`, next `3.8e-4` |
| affine `J`, Hamiltonian = Eq. (22), by monomials | `J(x) grad H(x)` can contain only quadratic monomials with a factor in {`electron_delta_n`, `B`}. **38 of the 69** quadratic monomials of `F` (65.6 percent of the coefficient norm) cannot be produced, all in the `C` equation: `(rho)(N)` 24 (coherence times `N`, from `-i N [g^T, rho]`), `(rho)(A)` 8 and `(rho)(rho)` 6 (squares of the coherence, from the bare Pauli source). | exact |

**Which Lie algebra.** None. The 31 coordinates are connected moments: `N`,
`A` and `C` are cumulants. Even the corresponding raw operators
`{E_ij, b_q, b_q^+ b_q', b_q b_q', E_ij b_q}` do not span a Lie algebra, as
one commutator shows:

```text
[E_ij b_q, E_kl b_r^+] = delta_qr E_ij E_kl + [E_ij, E_kl] b_r^+ b_q
```

The second term is an electron--two-boson operator `E b^+ b`. That is exactly
the third-order moment the archive factorizes in Eq. (14d), and the
factorization is where a Lie-Poisson structure would have been lost. The exact
moments obey the Lie-Poisson flow on the dual of the full, infinite operator
algebra with the linear Hamiltonian `<H>`. The 31-moment closure is not a
Poisson reduction of that flow.

**Polynomial first integrals of the drive-free flow.** The degree-two count
uses 2,500 random points and the degree-three count 9,000. The gap is at
least `5e12` in singular value.

- Degree at most two: 4. Degree at most three: 9.
- The nine are spanned by:
  - `ell`, `ell^2` and `ell^3`;
  - `H`;
  - the centre-mode coherent oscillator energy
    `|B_0 + B_1 + 2g/omega|^2`, with relative drift `2.5e-17`;
  - the products `ell H` and `ell |B_+ ...|^2`;
  - two genuinely cubic integrals, whose linear parts are the real and
    imaginary parts of `sum_q (C_q,01 + C_q,10)`.
- The span contains Eq. (22) to `5.2e-15`.
- With the drive on, 5 remain: the powers of `ell`, the centre-mode energy,
  and its product with `ell`.

**What bounds the correlation off-diagonals: nothing in this list.**

- `H` and the centre-mode energy do not contain them.
- The two cubic integrals contain them linearly at leading order. They fix
  only the symmetric sum `C_01 + C_10`, and a polynomial of odd degree is
  unbounded.
- This matches the instability report: the coordinates reaching `1e4` are
  exactly these entries. From the exact correlated preparation the event is at
  `t = 141.4059` (converged). From the exact `t = 60` state it is at `434.19`
  converged (the record's `433.8964` was the `rtol 1e-10` value).

**Reading.** The coordinator's inference from the trace identity, the energy
conservation and the skew-adjoint linear part to a Hamiltonian or Poisson
system does not survive the tests that can carry content. No constant or
affine Poisson structure exists, for any Hamiltonian. Volume preservation
plus one conserved quadratic energy is all the structure there is, together
with the eight further first integrals listed.

## (b) Koopman--von Neumann

**Generator.** For `dx/dt = F(t, x)` on R^31, `psi(x, t)` with `|psi|^2` the
Liouville density obeys `i d_t psi = L(t) psi`, where

```text
L(t) = -(i/2) sum_k { F_k(t, x), d/dx_k }
     = -i [ F(t, x) . grad + (1/2) div F ]
```

Here `div F = 0` identically, so

```text
L(t) = -i (F_0 + (A_1 + f(t) D) x + A_2 (x (x) x)) . grad
```

This is formally Hermitian for any smooth `F`, and its construction uses no
Hamiltonian structure. Under the working split, `B(t)` enters as a
prescribed classical coefficient (27 coordinates).

**Cost model, single trajectory from the exact correlated preparation.**

- **Domain.** On the closure trajectory the largest absolute coordinate is
  `1.50`, `1.68` and `2.11` and the widest coordinate range is `0.61`, `1.28`
  and `1.97` (to `t = 4`, `20`, `60`); box width `W ~ 4`.
  - The closure escapes at `t = 141.41`, so any fixed box fails past that.
- **Resolution.** A packet must stay within `epsilon` of the characteristic
  while the linearized flow stretches it by `Lambda(T)`. That needs grid
  spacing `h <= epsilon / (3 Lambda)` and `M = W/h` points per coordinate.
  - Instability report, bare trajectory from the exact correlated
    preparation, supremum over the window: `Lambda = 13.0`, `51.0`, `3,169`
    at `T = 4`, `20`, `60`.
  - My recomputation along the same trajectory reproduces the from-`t0`
    values `12.98`, `39.30` and `2,457`.
- **Generator norm.** `||L|| ~ sum_k max|F_k| * pi/h`, where
  `sum_k max|F_k|` along the trajectory is `9.99`, `17.9` and `31.2` (a lower
  bound over the box).

| horizon `T` | `Lambda` | points per coordinate at `epsilon = 1e-2` | logical qubits `31 ceil(log2 M)` | `T ||L||` | classical cost of the same trajectory |
|---|---|---|---|---|---|
| 4 | 13.0 | `1.6e4` | 434 | `4.9e5` | closure ODE, DOP853 `rtol 1e-12`: under 1 s |
| 20 | 51.0 | `6.1e4` | 496 | `1.7e7` | `1.3` s |
| 60 | 3,169 | `3.8e6` | 682 | `5.6e9` | `3.7` s |

At `epsilon = 1e-3`, add about 3.3 qubits per coordinate (103 in all) and
multiply `T ||L||` by 10. Reading out each coordinate's mean by amplitude
estimation to `epsilon` costs `O(W/epsilon)` repetitions, about `400` per
coordinate per output time at `1e-2`, times 31 coordinates.

**Representation.**

- On quantum hardware the representation is the `31 log2 M`-qubit grid
  register; neither a tensor network nor sampling is required.
- Any classical emulation or benchmark of the KvN route cannot hold the grid:
  `M^31 ~ 1e130` at `T = 4`. It would have to adopt a tensor-train or MPS
  representation of `psi` over 31 modes, or a sampled (particle)
  representation. A sampled representation of a point preparation is
  integrating the ODE itself.
- KvN transports a distribution: the mean of a finite packet is not the
  characteristic (`E F(X) != F(E X)`). The resolution rule above is the
  sufficient envelope from the B-Q2 dossier, not a necessity proof.

**Against the exact model.**

| | two sites | three sites |
|---|---|---|
| exact state dimension, cutoff 16 | 1,156 (= 4 x 17^2) | 44,217 (= 9 x 17^3) |
| classical exact propagation to `t = 20` | `3.06` s (record) | about `137` s (record, fitted law) |
| qubits for direct Hamiltonian simulation (one up and one down electron; 5 qubits per 17-level mode) | 12 | 19 |
| `||H_static||`, measured | `24.68` | `33.15` |
| drive operator norm | `1` | `1` |
| `T (||H|| + max|f|)` at `T = 20` | about `510` | about `680` |
| closure error | none | none |

The closure trajectory itself differs from exact by all-coordinate scalar RMS
`0.111`, `0.193` and `0.325` at `t = 4`, `20` and `60`.

**Verdict.** KvN is admissible in principle: unitary, with no dissipation or
Hamiltonian condition. For this target it is dominated on every axis:

- about 30 times the qubits of direct simulation of the exact model;
- `1e3` to `1e7` times the evolution-time-norm product;
- it inherits the closure's error and its escape;
- for a deterministic point preparation it reproduces what classical
  integration does in seconds.

## (c) Schrödingerization of the trajectory-linearized flow

**The flow.** `d xi/dt = J_F(t, x(t)) xi + s(t)`, with
`s = F(t, x) - dx/dt` along the chosen trajectory `x(t)`.

**Amplification**, `||Phi(T, 0)||_2` in the Euclidean closed coordinates, with
the supremum over `0 <= s <= t <= T` in brackets. The authority is the
instability report, table 5.1:

| trajectory | `T = 4` | `T = 20` | `T = 60` |
|---|---|---|---|
| bare closure from the exact correlated preparation | `12.98` | `39.3` (`51.0`) | `2,457` (`3,169`) |
| exact driven reference from the Hartree--Fock product state | `13.63` | `15.7` (`17.6`) | `34.7` (`54.4`) |
| exact driven reference from the correlated ground state (this round, not in the report) | `12.11` | `15.77` (`19.25`, 0.5 grid) | `26.30` (`36.91`) |

- My integration along the bare closure from the exact correlated
  preparation reproduces the report's row: `12.98`, `39.30`, `2,457`.
- `det Phi = 1` to `8e-9`, as the zero trace requires.
- The instability report's stacked backward-Euler condition numbers are
  `1.1e4` to `1.2e4` (`T = 4`), `7.9e4` to `2.0e5` (`T = 20`) and `5.2e5` to
  `1.16e7` (`T = 60`). Step count alone gives `2,038`, `10,187` and `30,559`.
  One step's condition number is at most `1.038`.

**Cost factor.** Jin, Liu and Ma (arXiv:2402.14696, Theorem 1) allow a
Hermitian part `H_1` with positive eigenvalues. Recovery needs the auxiliary
variable past `p >= T sup_t lambda_max^+(H_1(t))`, and amplitude amplification
retains `O(e^{p} ||u(0)|| / ||u(T)||)`. I report both that supremum form and
the integral `int_0^T lambda_max^+(H_1(t)) dt`, which is no larger. The
integral is my reading for a time-dependent generator, not their theorem.

| trajectory, metric | `int mu^+`, `T = 4` | `T = 20` | `T = 60` | `sup mu` |
|---|---|---|---|---|
| exact (correlated), Euclidean | `7.30` | `35.1` | `107.5` | `1.85` to `1.99` |
| bare closure from exact correlated, Euclidean | `7.11` | `37.3` | `115.8` | `1.85` to `2.45` |
| exact (correlated), metric `G` with `A_1` skew-adjoint | `51.2` | `250.9` | `786` | `14.3` to `17.1` |

**Reading.**

- **The overhead is far above the realized amplification.** The factor
  `e^{p}` is at least `e^{7.3} = 1.5e3` at `T = 4` and `e^{35} = 1.8e15` at
  `T = 20` in the Euclidean coordinates, while the realized amplification is
  only 12 to 51. The overhead is at least `e^{p} / Lambda`: about `1e2` at
  `T = 4` and about `1e14` at `T = 20`.
- **The skew-adjoint metric makes it worse.** The constant metric in which
  `A_1` is skew-adjoint raises the integral sevenfold.
- **There is no metric common to `A_1` and `D`**
  (`run_task2_metric.py`, `task2_metric.json`).
  - The symmetric `G` with `A_1` skew-adjoint form a 46-dimensional space.
    Adding `D` skew-adjoint cuts it to 8 dimensions.
  - At trace 31, the largest attainable smallest eigenvalue on those 8
    dimensions is `1.9e-8`, zero to SCS tolerance. No positive definite
    common metric exists.
  - The task-1 eigenvector metric fails `D` at relative `0.21`.
  - The driven linear propagator of `A_1 + f(t) D`, without `A_2` and priced
    directly to `t = 20`, has supremum `||Phi(t,s)||` of `12.09` in the
    Euclidean coordinates (undriven `exp(A_1 t)`: `12.08`; bound
    `sqrt(cond G) = 55.4`) and `7.23` in `G` (`1` undriven).
  - So the drive costs a factor of at most `7.2` in `G`, and almost nothing
    in Euclidean norm.
  - The growth from `13` to `2,457` over `T = 4` to `60` comes from the
    quadratic term, not the drive.
- **Reaching the realized amplification would need a time-dependent
  metric.** Only a similarity that tracks the instantaneous growth direction
  would bring `int mu^+` down toward `log Lambda`, and none is constructed
  here.
- **The linearization is an approximation of its own.** Along the exact
  trajectory, with `xi(0) = 0`, it reproduces the closure's departure from
  exact to relative `0.17` at `T = 4` (maximum error `0.121` against a
  maximum departure of `0.82`). At `T = 20` it is `1.07` and at `T = 60` it is
  `1.61`: the dropped `A_2(xi, xi)` term dominates once the departure is of
  order one.
- **Neither choice of base trajectory yields the closure trajectory.**
  Linearizing about the exact trajectory presupposes the exact trajectory as
  input. Linearizing about the closure's own trajectory gives `s = 0` and
  propagates perturbations, not the trajectory.

## The windowed re-lift, one table

Every `window` time units the order-`K` truncation is re-lifted from its own
first block. This assumes a free and exact readout of all 31 coordinates at
each restart, so it is a lower bound on any real windowed scheme's error.
Error is the maximum absolute coordinate error against the closure's own
trajectory from the exact correlated preparation; RK4 at `0.0025`. A dash
means the windowed run reached the `1e4` magnitude event before `t = 20`.

| `K` | window | restarts with a 31-coordinate readout (to `t = 20`) | `t <= 1` | `t <= 4` | `t <= 20` |
|---|---|---|---|---|---|
| 2 | 0.25 | 43, then event at `t = 11.0` | `3.2e-3` | `9.4e-2` | -- |
| 2 | 0.5 | 19, event at `10.0` | `1.4e-2` | `0.175` | -- |
| 2 | 1 | 10, event at `11.0` | `5.1e-2` | `0.344` | -- |
| 3 | 0.25 | 80 | `5.1e-5` | `2.6e-3` | `0.304` |
| 3 | 0.5 | 40 | `7.3e-4` | `1.4e-2` | `0.893` |
| 3 | 1 | 20 | `7.6e-3` | `9.5e-2` | `2.20` |
| 3 | 2 | 7, event at `16.0` | `7.6e-3` | `0.409` | -- |
| 4 | 0.5 | 40 | `2.4e-5` | `1.4e-3` | `0.503` |
| 4 | 1 | 20 | `1.2e-3` | `2.2e-2` | `1.02` |

Re-lifting the order-2 truncation compounds its error into the magnitude event
by `t = 10` to `12`.

## Reproduction (Task 2)

```bash
DIR=output/local_runs/paper_v_quantum_target_form_20261010
PYTHONPATH=paper_5/src $PY $DIR/run_task2_structure.py   # 59 s, 4.0 GB peak: energy, identities, invariants, constant-J test, monomial obstruction, Jacobi
PYTHONPATH=paper_5/src $PY $DIR/run_task2_identify.py    # names ell, the quadratic integrals and the rank-2 preserved tensor
PYTHONPATH=paper_5/src $PY $DIR/run_task2_affine.py      # 161 s, 3.8 GB: affine Poisson-tensor invariance nullspace
PYTHONPATH=paper_5/src $PY $DIR/run_task2_offdiag.py     # cubic integrals, centre-mode energy, off-diagonal involvement
PYTHONPATH=paper_5/src $PY $DIR/run_task2_dynamics.py    # 166 s, 2.1 GB: exact and closure to t=60, propagators, log-norms, linearized flow, ||H|| at L=2,3
PYTHONPATH=paper_5/src $PY $DIR/run_task2_metric.py      # common skew-adjoint metric for A_1 and D (SDP), driven linear propagator
PYTHONPATH=paper_5/src $PY $DIR/run_relift.py --orders 2 3
PYTHONPATH=paper_5/src $PY $DIR/run_relift.py --orders 4 --windows 0.5 1.0
```

---

# 2026-10-10, Task 3: B-Q3 architecture pass

The architecture, the source locks, the parity-test specifications, the
output contracts, the reduction table, the cost ledger and the seams are in
`agent_guidance/paper-v/handoffs/claude-architecture-quantum-integration-20261010.md`.
Recorded here are only its measured results.

**The propagated omitted tail is the recovered-coordinate error; its norm
certificate is loose.** `run_task3_tail.py`, `task3_tail_2.json`,
`task3_tail_3.json`.

The adjoint `W(s) = P_1 U_K(T, s)` was computed backward at `h = 0.005`, along
the bare closure's own trajectory from the exact correlated preparation.

| order `K` | horizon `T` | measured first-block error `||x(T) - X~_1(T)||` | relative mismatch of `int W tau ds` against it | norm certificate | certificate / measured | `sup_s ||P_1 U_K(T,s) P_K||` | terminal first-block weight |
|---|---|---|---|---|---|---|---|
| 2 | 4 | 2.246 | `2.6e-6` | 50.2 | 22 | 17.1 | 0.261 |
| 2 | 20 | 19.65 | `3.4e-6` | 2,886 | 147 | 142.7 | 0.182 |
| 3 | 4 | 2.045 | `3.9e-6` | 222.1 | 109 | 37.7 | 0.056 |
| 3 | 20 | 127.0 | `5.2e-6` | 83,506 | 657 | 2,746 | 0.032 |

- At order 4, from the stored runs, the terminal first-block weight is
  `0.0078` at `T = 4` and `0.0045` at `T = 20`.
- The pointwise tail ratio of Task 1 (median `0.72` at every `K`) predicts
  neither the measured error nor the weight. As the dossier said, it is a
  diagnostic, not a bound.

**Order-4 step refinement to `T = 20`**, requested by the B-Q2 dossier: RK4 at
`h = 0.005` against `0.0025` differs by at most `2.3e-6` in the first block,
against a truncation error of `387` (`check_steps_k4_t20.json`). The task-1
`K = 4, T = 20` entry stands.

**Reproduction (Task 3).**

```bash
PYTHONPATH=paper_5/src $PY $DIR/run_carleman.py --orders 4 --sample-stride 2 --tag _h0005_t20   # 140 s
$PY $DIR/check_steps_k4_t20.py
PYTHONPATH=paper_5/src $PY $DIR/run_task3_tail.py --orders 2   # 9 s
PYTHONPATH=paper_5/src $PY $DIR/run_task3_tail.py --orders 3   # 277 s
```

---

# Corrections after verification (2026-10-11)

**Scope.**

- **What was verified.** This report as committed at `a23ff642` (SHA-256
  `3c1a9afd...5849`) and the B-Q3 reply. The verifiers were four adversarial
  refuters (G1 structure, G2 Koopman and Schrödingerization, G3 tail and
  parity, G4 contract and seams) and a completeness critic. Their artifacts
  are in `output/local_runs/paper_v_quantum_verify_20261010/`.
- **How the earlier text is treated.** It is not rewritten. Where it
  conflicts with this section, this section governs.
- **Where the details are.** The calculations, receipts and the full
  correction table are in revision 2 of the reply,
  `agent_guidance/paper-v/handoffs/claude-architecture-quantum-integration-20261010-r2.md`.
  Receipts are under `output/local_runs/paper_v_quantum_target_form_20261010/r2/`.

## The one decision, restated

O3 at `epsilon = 1e-2` against the physical reference is out of reach for any
route that integrates the archive closure. The closure's own error against
the exact cutoff-16 reference is:

- site-0 occupation per spin: `0.028` (`T = 4`) and `0.41` (`T = 20`);
- Eq. (22) energy: `0.015` at both horizons.

The user's decision is the target O3 is scored against:

- **T-bare:** an integration-accuracy contract for the archive equations,
  with the model error on its own line. It is attainable at `1e-2` by the
  re-centred windowed lift (`K = 3`, window 1, 19 re-lifts with free exact
  readout): `2.6e-5` and `1.6e-4` on the site-0 occupation at `T = 4` and
  `20`.
- **T-physical:** attainable only by direct 12-qubit simulation of the exact
  dimer, `T (||H|| + max|f|) = 102` (`T = 4`) and `511` (`T = 20`), which
  leaves the archive equations out.

-> **Recommended: T-bare.**

## Corrections to this report's statements

**Task 1.**

- **The verdict.** Narrowed on 2026-10-10 to the Liu et al. condition and the
  tested origin-centred truncations. The windowed and re-centred results
  below confirm that the narrowing matters.
- **The drift on exact data.** "The closure diverges from exact data at
  `t = 433.8964`" is the `rtol 1e-10` value from the exact `t = 60` state;
  converged it is `434.19`. From the exact correlated preparation at
  `t = 0`, the `1e4` event is at `t = 141.4059` (instability report).
- **The cubic term.** It is zero both in the 31 closed coordinates and in the
  draft's own 44-real vectorization (`A_3 <= 7.4e-17`). There `F_0 = 0`
  exactly: the five nonzeros of `F_0` come from the affine closed chart
  `rho = I/2 + ...`. So the manuscript's constant-free force polynomial is
  right in its own coordinates, and only its cubic term is wrong. The module
  docstring is corrected.
- **The divergence-free statement.** It is stronger than stated: every
  diagonal entry `dF_i/dx_i` vanishes identically.

**Task 2(a).**

- **The linear part.** It preserves a **41**-dimensional space of constant
  antisymmetric tensors, not 39; the 39 was a column-scaling artifact.
- **What the negative results cover.** "No constant or affine Poisson tensor
  generates `F`" holds in the closed chart and under affine changes of it.
  The preserved affine family is `span{u^v, ell u^v, R^w, E^w}`, whose ranges
  lie in the 3-dimensional centre-mode subspace `(Re B_+, Im B_+, N_++)`.
- **The drive-free field is globally Hamiltonian.** Its Hamiltonian is the
  Eq. (22) energy, for the **degree-2** Poisson tensor `F^w / (2 omega)`, with
  `w = e_phonon00 + e_phonon11 + e_phonon01real`. Its Jacobi residual is
  `4.3e-16` and its generation residual `4.4e-16`. The rank-4 bivector
  `F^w + u^v` is also preserved.
- **Withdrawn.** "Volume preservation plus one conserved quadratic energy is
  all the structure there is" is withdrawn. The structure found has
  2-dimensional leaves and no evident computational use.
- **First integrals.** The nine have functional rank 5. They are all
  invariant under translations in a 6-dimensional subspace of the 8
  off-diagonal `C` coordinates, which is why none bounds them (this replaces
  the odd-degree argument).
- **Commutators.** The commutator non-closure was a hand derivation; G1
  verified it numerically (92 of 595 pairs leave the span).

**Task 2(b), Koopman--von Neumann.**

- **What the old figures were.** 434 to 682 qubits and `T ||L||` from
  `4.9e5` to `5.6e9` came from a heuristic (`h = epsilon/(3 Lambda)`,
  `W = 4`, generator norm sampled on the trajectory). That is neither a lower
  bound nor a sufficient cost.
- **Corrected figures:**
  - with the box generator norm, `T ||L|| >= 2.0e7`, `3.9e8`, `7.3e10`;
  - resolving the compressed direction needs 558, 682, 1,023 qubits;
  - the dossier's envelope needs 651, 1,984, 5,518.
- **The baseline.** The same-problem baseline is the dimer: 12 qubits and
  `T (||H|| + max|f|) = 102`, `511`, `1,533` at `T = 4, 20, 60`. The 19-qubit,
  680 figure belongs to the three-site chain.
- **Readout bias (new, r2 calculation 7).** The packet-mean readout is biased
  by about `sigma_0^2`. A bias at most `1e-2` needs `sigma_0 <= 0.020`
  (`T = 4`) and `<= 0.0024` (`T = 20`). At `sigma_0 = 0.1`, 86 of 2,000
  samples escape by `T = 20`.

**Task 2(c), Schrödingerization.**

- **"Realized amplification 12 to 51" mixed trajectories.** Along the exact
  trajectory the amplification is `12.1` and `15.8`, supremum `19.2`. Along
  T-bare the supremum over pairs is `13.0` and `51.0`, or `13.2` and `54.1`
  with the between-grid inflation and RK4 allowance (r2 calculation 3).
- **"Overhead at least `e^p/Lambda`" is replaced.** For the forced linearized
  flow the recovery factor is `1.2e3` to `1.5e3` at `T = 4` and `8.7e14` to
  `1.8e15` at `T = 20`. The theorem form `T sup mu` is `7.39` and `36.97`
  along the exact trajectory, and `7.38` and `46.13` along T-bare.
- **The metric is not unique, and a time-dependent one is not needed.** "The
  `A_1` metric" is one member of a 46-dimensional family, and the driven
  supremum in it is at least `7.27`. A **constant** metric fitted along
  T-bare (`sqrt(cond) = 8.45`) brings the T-bare exponent to `2.09` and
  `11.8` in the theorem form (r2 calculation 8).
- **No common metric.** No positive definite metric makes both `A_1` and `D`
  skew-adjoint. Proof: the Lie algebra they generate (dimension 91) contains a
  nilpotent `X` with `||exp(1000 X)|| = 647`. The same holds after the
  28-coordinate reduction (`679`).

**Windowed re-lift.**

- **The re-lift bound.** It is a bound only on **origin-centred** lifts, and
  "80 restarts" is 79 re-lifts plus the final readout.
- **The re-centred result.** Re-centred at each read-out state, `K = 3` with
  79 re-lifts reaches `4.1e-8` (`t <= 4`) and `6.3e-7` (`t <= 20`) in the
  largest coordinate error. With 19 re-lifts (window 1) it reaches `1.1e-4`
  and `1.8e-3`.

**Task 3.**

- **The tail identity.** It is Duhamel's formula, exact by construction,
  evaluated to quadrature accuracy: `2.6e-6` to `5.2e-6` with the trapezoid
  rule, `1.1e-9` to `4.1e-9` with Simpson. The agreement validates the
  pipeline.
- **The first-block errors.** They are `2.0` to `2.2` at `T = 4` and `19.6`
  to `127` at `T = 20`, not "of order one".
- **The terminal weights.** They are for the unscaled Kronecker encoding with
  the constant block excluded. At `gamma = ||x_0||` they are `0.525`, `0.249`,
  `0.080` at `T = 4`.
- **The pointwise tail ratio.** Its median along the closure trajectory is
  `0.547` (`t <= 4`) and `0.474` (`t <= 20`); `0.72` was on the exact states.

## New, measured in revision 2

- **Coefficient parity against Eqs. (14a)-(14e).** It is now tested. The
  Hamiltonian-exact rows agree with the exact cutoff-16 and cutoff-20
  derivatives to `<= 1.4e-14`, and a second transcription agrees on all 31
  rows to `5.9e-16`. Both catch the mutation `0.3 x_0 x_3`, including in row
  21. Tests: `paper_5/tests/test_archive_hamiltonian_parity.py`.
- **The reduced 28-coordinate system.** It fails every conserved-system
  condition:
  - all 4 zero modes receive the quadratic term;
  - there are 33 nonzero order-2 resonances;
  - no sector partition exists;
  - there is no common metric.

  Tests: `paper_5/tests/test_conserved_structure.py`.
