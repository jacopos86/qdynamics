# Claude architecture reply B-Q3, revision 2: corrections after verification

Date: 2026-10-11. Author: the persistent Paper V quantum-integration agent
(Claude).

**What this amends.** The committed reply
`agent_guidance/paper-v/handoffs/claude-architecture-quantum-integration-20261010.md`
at commit `a23ff642`, SHA-256
`85281ce683f382a63804b3e1b80987c3affa2427efc21e5fbc049dc729f1b74f`.

- That file is not edited. Every statement of it not amended here stands by
  reference.
- Line numbers below (`L..`) are that file's.

**What it answers.** The adversarial verification of that reply and of the
report:

- the scratchpad file `quantum_bq3_verify.json`: refuters G1-structure,
  G2-koopman-schrodingerization, G3-tail-and-parity, G4-contract-and-seams,
  and a completeness critic;
- their artifacts under
  `output/local_runs/paper_v_quantum_verify_20261010/`.

A refuted claim is taken as corrected by the refuter's corrected statement
unless I could show the refuter wrong from code or a rerun. Section 7 lists
the one place I add to a refuter. There is no place where I overturn one.

---

## Status (replaces L11, L21, L131-132, L343-344)

| item | value |
|---|---|
| branch, worktree | `claude/paper-v-routes-20260914`, `Holstein_test.worktrees/claude-paper-v-routes-20260914` |
| HEAD at writing | `840672b4` |
| reply as verified | committed at `a23ff642`, `85281ce683f382a63804b3e1b80987c3affa2427efc21e5fbc049dc729f1b74f` |
| report as verified | committed at `a23ff642`, `3c1a9afd9f76bdc8cf22738e544477b29c8a320164528feea53010af801c5849` |
| `quantum_target_form.py` as verified | committed at `a23ff642` (not "uncommitted", as L131 and L343-344 said), `1d8b520cfee33c956f0714d41ede829d14dcc5af136a70bb9d478eccb928773f` (L131's suffix was wrong) |
| `task1_form.npz` | `486605778ab5f292bbf739582643477445f2ad97f02c1759ec69c34a4f05d350` (L132's suffix was wrong) |
| this revision's uncommitted files | the hashes are in section 9 |
| run directory of this revision | `output/local_runs/paper_v_quantum_target_form_20261010/r2/` (gitignored); one receipt per calculation, `c1_*.json` to `c8_*.json` |

This agent ran no state-changing git command. Nothing is committed by this
agent and the manuscript is not edited.

---

## 0. First: what changes, and the one decision

**The recommendation of L110-115 is withdrawn.** It was O3 at
`epsilon = 1e-2` against the physical reference, and no route that integrates
the archive closure can meet that. The closure's own error against the exact
cutoff-16 reference is already above `1e-2` on both recommended observables
(`c5`, model-error line):

- site-0 occupation per spin: `0.028` at `T = 4` and `0.41` at `T = 20`;
- Eq. (22) energy: `0.015` at both horizons.

Against the bare closure (T-bare), which is the archive equations themselves,
O3 at `1e-2` is attainable. Only one route attains it: the lift re-centred
at each read-out state (section 2.1). The origin-centred lift does not
attain it at any order tested: the site-0 occupation error is `0.0915` at
order 4 at `T = 4`.

**The decision that is the user's.** O3 is selected observables at
`T = 4, 20` to `1e-2` absolute: the site-0 occupation per spin and the
Eq. (22) energy. It must be scored against one of two targets:

- **(a) T-bare.** This is an integration-accuracy contract for the archive
  equations themselves. The closure's model error (`0.015` to `0.41`) is
  reported on its own line and never folded in.
- **(b) T-physical.** This is attainable only by a route that does not
  integrate the closure: direct simulation of the exact dimer model on 12
  qubits, with `T (||H|| + max|f|) = 102` at `T = 4` and `511` at `T = 20`.
  That route drops the archive equations from the quantum claim.

-> **Recommend (a).** The draft's objective is the archive equations of
motion, so the claim should be about integrating them. The closure's model
error stays in the paper as a separate, already-measured line, and direct
simulation stays as the baseline every closure route must be compared
against.

---

## 1. Corrections, one line each, with the number relied on

| # | original (reply line) | corrected statement | source |
|---|---|---|---|
| 1 | L30-32, L210-211: the propagated tail is "measured exactly" and equals the error "to integration accuracy" | Duhamel's identity holds by construction; it was evaluated to quadrature accuracy. The trapezoid rule at `h = 0.005` gives `2.6e-6` to `5.2e-6` relative mismatch. That falls 4.0 times at `h = 0.0025`, and Simpson gives `1.1e-9` to `4.1e-9`. The agreement validates the adjoint pipeline; it is not a finding | G3 |
| 2 | L213-214: the error is "already of order one" | The Euclidean first-block error is `2.25` (`K = 2`) and `2.05` (`K = 3`) at `T = 4`, and `19.6` (`K = 2`) and `127` (`K = 3`) at `T = 20` | G3 |
| 3 | L33-34, L188-193: the certificate is loose 22 to 657 times | Stands as a sampled, a-posteriori quantity, since `tau_K` is evaluated on the trajectory being computed. The looseness splits into norm misalignment (`11.7` to `78`) and cancellation in time (`1.6` to `12.5`) | G3 |
| 4 | L35-37, L186, L215-218, L332, L338: terminal weight and postselection probability | Declare the encoding: Kronecker power, constant block excluded, `gamma = 1` (unscaled). The weights are `0.261`, `0.0561`, `0.00775` at `T = 4` (`K = 2, 3, 4`), and the postselection probability is their square: `0.068`, `3.2e-3`, `6.0e-5`. At `gamma = ||x_0|| = 2.283` the weights are `0.525`, `0.249`, `0.080` (measured again in `c5`); in the distinct-monomial encoding they are `0.347` and `0.111` (`K = 2, 3`). The weight trades against block normalization (section 4) | G3; `c5` |
| 5 | L38-39: the "median 0.72" pointwise tail ratio | `0.72` is on the exact states. Along the closure trajectory that section 3 uses, the median is `0.547` (`t <= 4`) and `0.474` (`t <= 20`), the same for every `K` | G3 |
| 6 | L148-161: "Independent verifications" | These were self-consistency checks against the same `closed_scalar_rhs`. A mutation `0.3 x_0 x_3` in row 21 passed every one of them. The named drive test checks two probe times, the skewness of `D`, and undriven parity at `t = 0`, where `f = 0`; it does not check random `(t, x)`. **Replaced by calculation 2** (section 2.2), which catches that mutation | G3; `c2` |
| 7 | L58-59: "the draft supports four contracts" | The draft anchors three: O1 (trajectory, as a small-system diagnostic, tex:320-324 and 443-445), O3 (selected observables, tex:443-445 and 461-463) and O4 (invariants and residuals, tex:322-324 and 443-445). O2 is my addition. tex:500-502 is a cost-model rule, not a contract. The anchors at tex:322-324 and 500-502 were paraphrases, not verbatim quotes | G4 |
| 8 | L58-61: requests the contract set missed | The draft also asks for: spectra by amplitude estimation (tex:336); a representation or compression claim (tex:305, 407); the classical characterization set `{dim x, nnz J_Q, spec J_Q, kappa(I - dt J_Q), Delta_phys}` (tex:395-403); solver rejection counts and post-pulse energy (tex:347); and the metric `M_ij` and force `V_i` measurements (tex:438-442). In the draft's split `B` is classical, so `B` is not a quantum output of O1 or O2 | G4 |
| 9 | L112-113: "O3 is the only contract whose readout does not scale with 31 N_t" | O2, O3 and O4 all have readout independent of `N_t`; only O1 scales with `31 N_t` | G4 |
| 10 | L97: the O3 functionals | `N00 + N11` is only the fluctuation occupation. The total phonon number `|B|^2 + Tr N` is quadratic, and `|B|^2` is `79` to `88` percent of it on the exact reference. "Block 2 of a lift supplies `x (x) x` directly" is false in a truncated lift: `||X~_2 - x (x) x||_F` is `11.3` to `802` (`K = 2, 3`; `T = 4, 20`). The energy read from block 2 misses T-bare by `0.178` (`K = 2`), `0.029` (`K = 3`) and `0.0094` (`K = 4`, `T = 4`) | G4; `c1` |
| 11 | L320-326: "no linear map from `X_K` evaluates the repair's cones" | **Withdrawn for the retained joint Gram.** Its entries are exact degree-2 polynomials of the 31 coordinates (cubic coefficient `0.0`, quadratic only in the 3 `rho` coordinates, reconstruction `3.1e-16`), so a linear map from an exact `(1, X_1, X_2)` evaluates it. The separation that holds is narrower: block 2 of a *truncated* lift is not `x (x) x` (row 10); cone membership is a spectral condition, not a linear one; and only the positive extension's auxiliary rows lie outside the 31 coordinates | critic C1 |
| 12 | L334: kappa_stack "1.1e4 to 1.16e7" | This mixed targets. For T-bare from the exact correlated preparation, read from `cond_exact_correlated_t0_t0_0.json` (SHA-256 `d3485acc...579eb`), it is `12,071`, `166,189` and `11,558,518` at `T = 4, 20, 60`, against `2,038`, `10,187` and `30,559` from the step count alone | G4; `c5` |
| 13 | L335: the Newton or feedback row, "0.17 at T = 4, 1.07 at T = 20" | Those numbers linearize about the exact trajectory. That puts an exact reference in a propagation path, so it is inadmissible as a route and is labelled T-physical-input diagnostic only. A model-error line is added (section 4) | G4 |
| 14 | L44-46, L263-267, L273: the only preserved tensors are rank 2 on the centre mode, and there is no Darboux or Lie-Poisson map | This holds only for **constant and affine** bivectors **in the closed chart** (or any affine change of it). The preserved affine family is `span{u^v, ell u^v, R^w, E^w}`, with ranges in the 3-dimensional centre-mode subspace `(Re B_+, Im B_+, N_++)`, so rank at most 2 is automatic there. `w = e_phonon00 + e_phonon11 + e_phonon01real` is a constant symmetry: `A_1 w = A_2(., w) = D w = 0` exactly. With degree-2 coefficients the flow also preserves `F^w + u^v` (rank 4) and `F^w`. The drive-free `F` **is globally Hamiltonian**, with the Eq. (22) energy, for the degree-2 Poisson tensor `F^w / (2 omega)`: Jacobi residual `4.3e-16`, generation residual `4.4e-16`. That structure has 2-dimensional leaves and no evident computational use. Lie-Poisson structures affine in another chart, such as the raw moments, are not covered | G1 |
| 15 | L44-46: "the only exact reductions" | Restricted to **linear** reductions. The nine polynomial first integrals have functional rank 5 (`ell`, `H`, `E_c`, and two cubic ones), so nonlinear level-set reductions exist and are not assessed. All nine are invariant under translations in a 6-dimensional subspace of the 8 off-diagonal `C` coordinates, which is the valid reason no function of them bounds those coordinates (it replaces "a polynomial of odd degree is unbounded") | G1 |
| 16 | report Task 2(a): "the linear part preserves a 39-dimensional family" | It is **41**. The 39 came from column scaling of two rounding-level columns in `nullspace_report` | G1 |
| 17 | L281: status on the reduced system | Replaced by reduced-system numbers (calculation 4, section 2.4): rank `L = 24`, rank `[F_0'|L] = 24`, rank `[F_0'|L|A_2'] = 28`; all 4 of the 4 left zero modes receive a nonzero quadratic projection (singular values `1.09`, `0.84`, `0.82`, `0.78`) | critic; `c4` |
| 18 | L284: nonresonance "fails automatically at order three" | It fails at **order two**, with nonzero coefficients: 33 of the 79 ordered resonant pairs carry nonzero quadratic coupling (smallest `0.022` reduced, `0.031` full; largest vanishing one `1.1e-15`), and 29 of them involve a zero eigenvalue | critic; `c4` |
| 19 | L50, L286: no common metric, from the SCS value `1.9e-8` | Replaced by a proof. The Lie algebra generated by `A_1` and `D` has dimension 91 and contains `X` with `||X^2||_F = 2.4e-13` (nilpotent of index 2), so `||exp(t X)||` grows linearly (`647` at `t = 1000`). A positive `G` making `A_1` and `D` skew would make `exp(t X)` bounded by `sqrt(cond G)`. After the reduction the same holds: dimension 91, `||X^2|| = 4.4e-13`, `||exp(1000 X)|| = 679` | G2; `c4` |
| 20 | L52, L286: "7.23 in the `A_1` metric (undriven 1)" | "The `A_1` metric" is not unique: the task-1 eigenvector metric is one point of a 46-dimensional `A_1`-skew family. In it the driven supremum is at least `7.27` (grid maximum on 0.05 grids); in another member, `G*` (condition number 561), it is `6.21`. Both are sampled lower bounds | G2 |
| 21 | L53-57: KvN "430 to 680 qubits, `T ||L||` from 5e5 to 6e9" | This is a heuristic (`h = epsilon/(3 Lambda)`, `W = 4`, generator norm sampled on the trajectory), neither a lower bound nor a sufficient cost. With the box generator norm, `T ||L|| >= 2.0e7`, `3.9e8`, `7.3e10`. Resolving the compressed direction needs `558`, `682`, `1,023` qubits; the dossier's envelope needs `651`, `1,984`, `5,518` (`T = 4, 20, 60`). Calculation 7 adds the readout bias (section 2.7) | G2; `c7` |
| 22 | L55-56: "direct simulation needs 12 or 19 qubits, `T ||H||` 510 to 680" | The baseline was the wrong problem: 19 qubits and 680 belong to the three-site chain. For the dimer at cutoff 16 direct simulation needs 12 qubits (11 dense), with `T (||H|| + max|f|) = 102`, `511`, `1,533` at `T = 4, 20, 60`. KvN therefore needs 36 to 57 times the qubits and `4.8e3` to `3.7e6` times the time-norm product at matched horizons | G2 |
| 23 | report Task 2(c): "realized amplification 12 to 51", "overhead at least `e^p/Lambda` (`1e2`, `1e14`)", "a time-dependent metric would be needed" | The first range mixed trajectories: along the exact trajectory the amplification is `12.1` (`T = 4`) and `15.8` (`T = 20`, supremum `19.2`). For the forced linearized flow the recovery factor is `1.2e3` to `1.5e3` at `T = 4` and `8.7e14` to `1.8e15` at `T = 20`. The theorem form `T sup mu` is `7.39` and `36.97`, beside the integral form `7.30` and `35.12`. A **constant** metric with `sqrt(cond) = 7.7` already gives exponents `1.4` and `6.9`, so a time-dependent metric is not required (calculation 8 measures this along T-bare) | G2; `c8` |
| 24 | report: the windowed re-lift "is a lower bound on any real windowed scheme" | This holds only for **origin-centred** lifts. Re-centred at each read-out state, the same 79 re-lifts reach `4.1e-8` (`t <= 4`) and `6.3e-7` (`t <= 20`) at `K = 3`. "80 restarts" is 79 re-lifts plus the final readout | G2; `c1` |
| 25 | L371-384 (section 7): "certified Lambda" from the terminal adjoint | This computed the wrong supremum and duplicated report Task 2(c). The tube needs the supremum over pairs `0 <= s <= t <= T`. **Replaced by calculation 3** (section 2.3) | G4; `c3` |
| 26 | report (Task 2 decision for the user): the target restated as `F_0 + (A_1 + f(t) D) x + A_2 (x (x) x)` | `F_0` is nonzero only in the affine closed chart (`rho = I/2 + ...`). In the draft's own real-doubled vectorization (44 reals including `B`), `F_0 = 0` and `A_3 <= 7.4e-17`. The manuscript's constant-free force polynomial is correct in its own coordinates; only its cubic term is zero. The module docstring is corrected (it said degree three) | G4 |
| 27 | report: the commutator statement | It was a hand derivation with no computation and concerns the span of the raw operators, since cumulants are not operators. Verified numerically by G1: 92 of 595 pairs leave the span on the truncation-free block | G1 |
| 28 | report: the divergence-free statement | It is stronger: every diagonal Jacobian entry `dF_i/dx_i` vanishes identically, and central differences of `closed_scalar_rhs` give `0.0` | G1, G3 |

---

## 2. The eight calculations

Each process ran in under 20 minutes. Every number in this section is on
T-bare unless it carries another label. T-bare is the bare closure from the
exact correlated preparation (cutoff-16 ground state), DOP853 at
`rtol 1e-12`. `epsilon = 1e-2` absolute.

### 2.1 O3 precision attainable against T-bare, per route (`c1_o3_precision.py`; `c1_origin.json`, `c1_k4.json`, `c1_recentred.json`)

Observables:

- **site0:** `(1 + x_0)/2`.
- **Hlin:** `h . x`, with `h` the gradient of the Eq. (22) energy at the
  origin.
- **H:** the Eq. (22) energy.

Error is `|O(route at T) - O(T-bare at T)|`.

**Origin-centred order-`K` lift** (`gamma = 1`). The first block uses RK4 at
`h = 0.0025`. `H` is read from block 1 by evaluating `H(X~_1)`, and from
block 2 as `h . X~_1 + Q_2 : X~_2`.

| `K` | site0 `T=4` | site0 `T=20` | Hlin `T=4` | Hlin `T=20` | H from block 1, `T=4` | H from block 2, `T=4` | H from block 2, `T=20` |
|---|---|---|---|---|---|---|---|
| 1 | 0.179 | 0.152 | 0.174 | 0.393 | 0.574 | -- | -- |
| 2 | 0.290 | 1.02 | 0.637 | 1.49 | 0.543 | 0.178 | 0.178 |
| 3 | 0.132 | 18.9 | 0.980 | 0.138 | 0.855 | 0.0290 | 0.0290 |
| 4 | 0.0915 | 194 | 0.446 | 79.5 | 0.568 | **0.0094** | not run |

At order 4 and `T = 4` there is also the single-covector adjoint
`c^T P_1 U_4(T,s)`, backward RK4 at `h = 0.005` on the transposed sparse
generator (`954,304` dimensions, `12.7e6` nonzeros):

- Duhamel identity (Simpson) against the measured error: `2.5e-9` (site0) and
  `2.3e-9` (Hlin) relative.
- Scalar bound `int |w^T P_4 tau_4| ds` over the measured error: `3.34`
  (site0) and `2.10` (Hlin).

**Re-centred windowed lift.** At each window start the order-`K` truncation of
`y = x - x_r` is built about the read-out state `x_r`, with forcing
`F(t, x_r)`, from `y = 0`. Readout is assumed free and exact.

The per-window first-block weight is measured at each window end, as a
median over windows. The first form excludes the constant block,
`||Y_1|| / ||(Y_1..Y_K)||`. The second includes the constant block of the
homogeneous embedding, `||Y_1|| / ||(1, Y_1..Y_K)||`.

| `K` | window | re-lifts (31-coordinate readouts) | site0 `T=4` | site0 `T=20` | H `T=4` | H `T=20` | max coordinate error `t<=4` / `t<=20` | weight, without / with constant block (median) |
|---|---|---|---|---|---|---|---|---|
| 2 | 0.25 | 79 | `3.5e-6` | `4.4e-5` | `6.5e-6` | `9.0e-7` | `2.2e-5` / `1.8e-4` | 0.94 / 0.34 (min 0.10) |
| 2 | 1 | 19 | `1.7e-3` | `1.0e-2` | `2.4e-3` | `1.1e-4` | `5.4e-3` / `4.5e-2` | 0.66 / 0.56 |
| 2 | 2 | 9 | `1.4e-2` | `0.238` | `7.2e-3` | `3.6e-2` | `6.3e-2` / `0.89` | 0.57 / 0.52 |
| 3 | 0.25 | 79 | `8.0e-9` | `2.2e-8` | `4.6e-9` | `4.4e-8` | `4.1e-8` / `6.3e-7` | 0.93 / 0.34 (min 0.10) |
| 3 | 1 | 19 | `2.6e-5` | `1.6e-4` | `8.4e-5` | `1.2e-4` | `1.1e-4` / `1.8e-3` | 0.51 / 0.45 |
| 3 | 2 | 9 | `2.3e-4` | `2.9e-3` | `6.6e-3` | `1.6e-2` | `7.6e-3` / `8.8e-2` | 0.38 / 0.37 |

**Reading.**

- **Origin-centred lift: never meets O3.** It meets `1e-2` only for `H` read
  from block 2 at order 4 and `T = 4` (`0.0094`). It never meets it for the
  site-0 occupation.
- **Re-centred windowed lift: meets O3 against T-bare.** It meets `1e-2` on
  both observables at both horizons at `K = 3` with windows up to 1 (19
  re-lifts).
- **What the re-centred figures assume.** They assume an exact readout of all
  31 coordinates at every restart, and a readout is a quantum
  state-tomography-scale cost. The next calculation (section 8) prices that
  readout's error.

### 2.2 Coefficient parity against the Hamiltonian and against Eqs. (14a)-(14e) (`paper_5/tests/test_archive_hamiltonian_parity.py`, 11 tests; receipt `c2_parity_receipt.json`)

**Hamiltonian parity.**

- **Setup.** At 81 exact driven states (`t = 0..4`, 0.05 grid), the exact
  contracted derivative of the closed coordinates is computed from the
  wavefunction derivative, with no archive code. `closed_scalar_rhs` is
  subtracted from it.
- **The six exact Ehrenfest rows** (the 3 `rho` rows and the real
  shared-trace and diagonal-difference rows of `C`): at most `1.0e-14`
  (cutoff 16) and `1.4e-14` (cutoff 20). The test asserts `1e-13`.
- **The B, N, A rows:** at most `1.2e-4`, `4.3e-5`, `7.3e-4` at cutoff 16,
  falling to `1.8e-6`, `1.2e-6`, `1.3e-5` at cutoff 20. That is the
  phonon-cutoff remainder; the test asserts a fall of at least 20 times.
- **The 11 factorized `C` rows** carry a closure defect of up to `0.258`
  against the Hamiltonian, so the Hamiltonian cannot check them.
- **Mutation `+0.3 x_0 x_3`:** it leaves a defect of `0.133` in each covered
  row tested (rows 0, 3, 7, 11, 17, 19). In row 21 it moves the defect from
  `0.121` to `0.175`, which the Hamiltonian check cannot discriminate.

**Second transcription.**

- **What it is.** `transcribed_closed_rhs` in the test file is Eqs.
  (14a)-(14e) and (15), hand-coded from the draft. It uses general coupling
  tensors `g^{q+-}_{12}`, explicit index sums, and spin degeneracy 2 in
  Eqs. (14b), (14c) and (14e).
- **Agreement.** It matches `closed_scalar_rhs` on all 31 rows to `5.9e-16`
  relative, at 60 random `(t, x)`.
- **Coefficients.** Its extracted coefficients equal the task-1 tensors:
  `F_0`, `A_1` and `D` exactly (`0.0`), and `A_2` to `1.4e-17`.
- **Mutation.** The row-21 mutation leaves a defect of `0.159`. The test
  asserts `> 1e-3`, and the test fails the mutated module as required.
- **Limits.** The second transcription shares with the first:
  - the coordinate maps `closed_scalar_to_matrix_state` and
    `matrix_state_to_closed_scalar_coordinates`;
  - the spin-degeneracy convention;
  - the drive convention `v_ext = diag(f/2, -f/2)`.

  So it is an independent hand-coding of the same equations, not an
  independent derivation of them.

### 2.3 Lambda over pairs, with inflation (`c3_pair_lambda.py`, `c3_pair_lambda.json`)

**Definition.** `Lambda(T) = sup over 0 <= s <= t <= T of ||Phi(t, s)||_2`,
for `J(t) = A_1 + f(t) D + 2 A_2(x(t), .)` along T-bare.

**Method.**

- `Phi(t, 0)` by RK4 at `h = 0.005` and at `h/2`, with stage states from the
  dense output.
- The grid supremum is taken from the `h/2` run. The RK4 allowance is the
  maximum pairwise difference between the two runs, divided by 15
  (Richardson).
- Between-grid inflation `exp(ds mu_bar) exp(0.005 mu_bar)`, with
  `mu_bar = max mu_2 + 2 ||A_2|| dt max||F|| + ||D|| dt max|f'|`, where
  `dt = 0.0025`, `||A_2|| = 1.995`, `||D|| = 1` and `max|f'| = pi/2`.

| `T` | start grid / end grid | grid supremum (where) | RK4 allowance | `max mu_2` | `max ||F||` | inflation | **Lambda** | admissible `delta_avg` | terminal `sup_s ||Phi(T,s)||` (the wrong quantity) and its `delta_avg` |
|---|---|---|---|---|---|---|---|---|---|
| 4 | 0.005 / 0.005 | 12.976 (`s = 0`, `t = 4`) | `3.2e-11` | 1.846 | 1.537 | 1.0188 | **13.22** | **`4.48e-5`** | 12.976, `4.65e-5` |
| 20 | 0.02 / 0.005 | 51.035 (`s = 0`, `t = 18.235`) | `1.9e-8` | 2.307 | 2.199 | 1.0600 | **54.10** | **`1.07e-7`** | 39.30, `2.03e-7` (1.9 times too permissive) |

Here `delta_avg = 1/(4 ||A_2|| Lambda^2 T^2)`. It is the largest
time-averaged defect for which the tube inequality `r <= a + b r^2` has a
solution, with radius at most `2 Lambda T delta_avg`.

Assumptions are stated and not bounded:

- the DOP853 base trajectory is taken as exact;
- `max ||F||` is a sampled maximum;
- the RK4 allowance is a Richardson estimate.

So this is "certified modulo those three", not a certificate.

### 2.4 Reduced system and the conserved-system conditions (`paper5/stability/conserved_structure.py`, `paper_5/tests/test_conserved_structure.py`, 6 tests; receipt `c4_reduced_system.json`)

**The reductions (map and inverse in the module).**

- **Centre mode.** `B_+ = B_0 + B_1` decouples exactly: in its rows, `A_1`
  on other coordinates, `D` and `A_2` are all `0.0`, and
  `F_0 = (0, -2g) = (0, -1.2247)`. So `dB_+/dt = -i omega B_+ - 2 i g`.
- **The linear first integral.** `ell = s_r + (g/6)(N00 + N11) + (g/3) Re N01`.
  The smallest singular value of `[F_0|A_1|A_2|D]` is `1.8e-15`, and the next
  is `0.468`.
- **The 28-coordinate system.** `B_+` is fixed at `-2g/omega` and `s_r` is
  eliminated by `ell`. This reproduces the full T-bare trajectory to
  `1.5e-4` over `t <= 20`, which is half the `B_+` oscillation amplitude
  `3.1e-4` (the cutoff-16 preparation sits that far from the centre-mode
  fixed point).

**The conditions.**

| condition | full (31) | reduced (28) |
|---|---|---|
| rank `L`, rank `[F_0|L]`, rank `[F_0|L|A_2]` | 26, 26, 30 | 24, 24, 28 |
| forcing projection on the left null space | `1.4e-15` | `4.3e-16` |
| quadratic projection on the left zero modes (singular values) | `1.24, 1.12, 1.03, 0.94, 3e-16` (the fifth is `ell`) | `1.09, 0.84, 0.82, 0.78` (all four) |
| trace; largest diagonal entry; largest `|Re lambda|` | `0`, `0`, `2.2e-16` | `3.8e-17`, `2.0e-16`, `5.4e-16` |
| order-2 resonant ordered pairs / with nonzero coupling / involving a zero eigenvalue | 79 / 33 / 29 | 79 / 33 / 29 |
| symmetric `G` with `L` skew-adjoint / with `L` and `D` both skew-adjoint | 46 / 8 | 34 / 4 |
| Lie algebra of `{L, D}`: dimension; nilpotent element `||X^2||`; `||exp(1000 X)||` | 91; `2.4e-13`; 647 | 91; `4.4e-13`; 679 |
| signed sectors: modes fitting neither `S+` nor `S-` | 5 | 4 |
| share of the eigenbasis quadratic tensor outside the allowed blocks `P+ F2 (P+ (x) P+) + P- F2 (P- (x) P-)` | 0.89 | 1.00 |

**Reading.** On the 28-coordinate system no condition of Jennings et al.
Theorem 4.1, Corollary 4.3 or Theorem 5.12 is met:

- the neutral modes receive the nonlinearity;
- there is no strictly stable complement (trace zero);
- there is no admissible sector partition;
- nonresonance fails at order two;
- there is no common metric with the drive.

### 2.5 Ledger numbers, with target labels (`c5_ledger.py`, `c5_ledger.json`)

The block normalization of `G = L + f_max M` (`f_max = 0.8664`) is for the
order-`K` lift of `y = x/gamma`. All rows are T-bare. The products are
`alpha T / w`, where `w` is the terminal first-block weight (Kronecker
encoding, constant block excluded). The order-4 block norms come from the G3
forward run `fwd_K4_h0.005.npz` (SHA-256 recorded in the receipt).

| `gamma` | `K` | dimension | nonzeros | `s x max|entry|` | LCU bound | `||G||_2` | `w(4)` | `w(20)` | `alpha_LCU 4/w(4)` | `||G|| 4/w(4)` | `alpha_LCU 20/w(20)` |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2 | 992 | 6,549 | 200 | 15.6 | 6.65 | 0.261 | 0.182 | 240 | 102 | 1,720 |
| 1 | 3 | 30,783 | 310,405 | 1,660 | 23.4 | 9.73 | 0.0561 | 0.0321 | 1,670 | 693 | 14,600 |
| 1 | 4 | 954,304 | 12,948,144 | 3,324 | 31.3 | 12.86 | 0.00775 | 0.00449 | 16,100 | 6,640 | 139,000 |
| 2.283 | 2 | 992 | 6,549 | 200 | 19.7 | 6.90 | 0.525 | 0.389 | 150 | 53 | 1,020 |
| 2.283 | 3 | 30,783 | 310,405 | 1,660 | 29.6 | 9.88 | 0.249 | 0.153 | 476 | 159 | 3,880 |
| 2.283 | 4 | 954,304 | 12,948,144 | 3,324 | 39.5 | 13.22 | 0.0804 | 0.0477 | 1,960 | 658 | 16,600 |

**Reading.**

- **Rescaling helps.** Rescaling by `||x_0||` lowers `alpha T / w` by 1.6
  to 10 times despite the larger `A_2` coupling.
- **Order costs more than it buys.** Every order step multiplies the product
  by about 3 to 10. Order 4 buys accuracy only for `t <= 2` (Task 1).

Other ledger numbers:

- **kappa_stack (T-bare):** `12,071`, `166,189`, `11,558,518` at
  `T = 4, 20, 60`.
- **Model error (T-bare against T-physical), pointwise at `T` and as a sample
  RMS over `t <= T`:**

  | observable | at `T = 4` | at `T = 20` | sample RMS, `t <= 4` | sample RMS, `t <= 20` |
  |---|---|---|---|---|
  | site0 | 0.0284 | 0.412 | 0.0296 | 0.143 |
  | H | 0.0150 | 0.0150 | 0.0276 | 0.0182 |

### 2.6 Sign, scale and confidence per contract (`c6_sign_scale_confidence.py`, `c6_sign_scale_confidence.json`)

**Sign.**

- `x_0/2` is the part of the site-0 occupation an encoded vector carries; the
  constant `1/2` is not in it. It changes sign 2 times for `t <= 4` and 7
  times for `t <= 20` along T-bare, and 2 and 2 along T-physical. It passes
  through `0` at `t = 0` (`2.7e-15`).
- A relative-precision readout of it therefore fails near each crossing. It
  must be read to absolute precision `epsilon` with a sign reference, for
  example by interference with a known reference vector.
- The site-0 occupation itself, `Hlin`, `H` and `|B|^2 + Tr N` never change
  sign. Their minimum `|value|` over `t <= 20` on T-bare is `0.225`, `4.69`,
  `3.43` and `3.68`.

**Scale.**

- `||x(T)||` on T-bare is `2.283`, `2.281`, `2.337` at `T = 0, 4, 20`.
- The Kronecker lifted norm (`gamma = 1`) is:

  | `K` | `T = 4` | `T = 20` |
  |---|---|---|
  | 2 | 11.9 | 109 |
  | 3 | 55.7 | 3,939 |
  | 4 | 370 | 195,473 |

- An overlap `c . x` read from `|x>/||x||` needs about
  `pi ||c|| ||x|| / epsilon` amplitude-estimation calls: `358` (site0) and
  `3,826` (Hlin, `||h|| = 5.34`) at `T = 4`.

**Confidence.** Estimates per contract, and the union-bound factor
`ln(N/0.05)` at a total failure probability of `0.05`:

| contract | estimates | `ln(N/0.05)` |
|---|---|---|
| O1 (all 31 coordinates on the 0.0025 grid to `t = 20`) | 248,031 | 15.4 |
| O2 | 62 | 7.1 |
| O3 (two observables, two horizons) | 4 | 4.4 |
| O4 | 4 | 4.4 |

### 2.7 KvN readout bias by Monte Carlo (`c7_kvn_bias.py`, `c7_kvn_bias.json`)

- **Setup.** The packet is `X_0 ~ N(x_0, sigma_0^2 I)` in the 31 closed
  coordinates, with 2,000 samples taken as 1,000 antithetic pairs.
  Integration is batched RK4 at `h = 0.0025`; the batched field equals
  `closed_scalar_rhs` to `1.8e-15`.
- **Bias.** The bias is `||E X(T) - x(T)||`, with the Monte Carlo standard
  error in brackets.

| `sigma_0` | bias `T=4` | bias `T=20` | site0 readout bias `T=4` / `T=20` | H readout bias | packet spread RMS / `sigma_0` at `T = 4` / `20` | samples escaping by `T = 20` |
|---|---|---|---|---|---|---|
| `1e-3` | `2.6e-5` (`1.6e-6`) | `2.8e-3` (`1.3e-4`) | `7.5e-7` / `-4.4e-4` | `2.4e-6` | 18.1 / 51.1 | 0 |
| `1e-2` | `2.6e-3` (`1.6e-4`) | `0.178` (`6.6e-3`) | `7.7e-5` / `-2.2e-2` | `2.5e-4` | 18.5 / 50.5 | 0 |
| `1e-1` | `0.258` (`1.4e-2`) | `20.7` (`10.6`) | `7.3e-3` / `-0.118` | `2.4e-2` | 18.7 / 5,781 | 86 of 2,000 |

**Reading.**

- **Scaling.** The bias scales as `sigma_0^2`, as expected for a quadratic
  field. For a bias at most `1e-2`, the packet must satisfy
  `sigma_0 <= 0.020` at `T = 4` and `sigma_0 <= 0.0024` at `T = 20`. At
  `sigma_0 = 0.1`, 86 of 2,000 samples escape by `T = 20`.
- **What the KvN grid must resolve.** The packet's width, not only the
  coordinate precision: about `18` to `51` times `sigma_0`, stretched.

### 2.8 The propagator a trajectory-linearized route integrates (`c8_driven_linearized_propagator.py`, `c8_driven_linearized_propagator.json`)

The generator is `J(t) = A_1 + f(t) D + 2 A_2(x_bare(t), .)` along T-bare,
with `Phi` by RK4 at `h = 0.005`. Suprema are over `s <= t`, with `s` and `t`
on 0.05 grids; they are grid maxima, so lower bounds (section 2.3 gives the
inflated Euclidean value).

The three metrics:

- **G2:** refitted here by the G2 refuter's own SDP bisection, `I <= G <= 100 I`
  on Jacobians every 0.2 along the exact trajectory. It reproduces G2's
  `sqrt(cond G) = 7.719` and bisection value `0.3414` to every printed digit.
- **bare fit:** the same construction on T-bare's own Jacobians.
- **reduced 28:** the reduced system's propagator along the reduced
  trajectory.

The Schrödingerization exponent is given in the integral form
`int max(mu_G, 0) dt` (my reading) and the theorem form `T sup mu_G`.

| metric | `sqrt(cond G)` | sup `||Phi||_G`, `T=4` | sup `||Phi||_G`, `T=20` | `||Phi(20,0)||_G` | exponent `T=4` (integral / theorem) | exponent `T=20` (integral / theorem) |
|---|---|---|---|---|---|---|
| Euclidean | 1 | 12.98 | 51.03 | 39.30 | 7.11 / 7.38 | 37.26 / 46.13 |
| G2 (fitted along exact) | 7.72 | 3.24 | 28.5 | 26.1 | 2.18 / 3.08 | 12.52 / 17.91 |
| bare fit | 8.45 | 3.73 | 30.1 | 29.2 | 1.86 / 2.09 | 9.50 / 11.77 |
| reduced 28, Euclidean | 1 | 10.53 | 50.10 | 40.05 | -- | -- |

**Reading.**

- **A constant metric cuts the recovery factor by orders of magnitude.** The
  Euclidean theorem-form factor `e^{p}` is `e^{7.38} = 1.6e3` at `T = 4` and
  `e^{46.1} = 1.1e20` at `T = 20`. The bare-fit metric gives
  `e^{2.09} = 8.1` and `e^{11.8} = 1.3e5`, at a conversion cost of
  `sqrt(cond G) = 8.45` each way.
- **G2's exponents do not carry over from the exact trajectory to T-bare.**
  Its exponents along the exact trajectory (`1.4`, `6.9`) become `2.2` and
  `12.5` along T-bare.
- **The reduction does not change the growth.** The 28-coordinate propagator
  grows like the full one (`50.1` against `51.0` at `T = 20`).

---

## 3. Output contract (replaces L58-115)

| contract | draft anchor | object | target it can be scored against at `epsilon = 1e-2` | readout scaling |
|---|---|---|---|---|
| O1 trajectory, small-system diagnostic | tex:320-324, 443-445 | 27 quantum-target coordinates (`B` classical in the split) on a time grid | T-bare (classical integration already does this in about 1 s) | `27 N_t` estimates (`216,027` on the 0.0025 grid to `t = 20`; `248,031` if `B` is read too), `ln` factor 15.3 to 15.4 |
| O2 terminal state (my addition, no anchor) | none | 27 coordinates at `T` | T-bare | `27 x 2` |
| O3 selected observables | tex:443-445, 461-463 | site0 and `H` at `T = 4, 20` | T-bare: yes, by the re-centred windowed lift (section 2.1). T-physical: no closure-integrating route; only direct exact-model simulation | 4 estimates; site0 needs a sign reference for its encoded part `x_0/2` (section 2.6) |
| O4 invariants and residuals | tex:322-324, 443-445 | Eq. (22) drift, residual norms | T-bare | 4 estimates |
| not covered by O1 to O4 | tex:305, 336, 347, 395-403, 438-442 | spectra by amplitude estimation; a representation or compression claim; the characterization set; rejection counts and post-pulse energy; `M_ij` and `V_i` | -- | -- |

The decision is in section 0.

## 4. Ledger (replaces L328-339), every number with its target

| line | target | number |
|---|---|---|
| model (closure) error | T-bare against T-physical | site0 `0.028` (`T=4`), `0.41` (`T=20`); H `0.015`, `0.015` (section 2.5) |
| preparation | T-bare lift | initial first-block weight `0.401`, `0.173`, `0.076` (`K = 2, 3, 4`; `gamma = 1`, Kronecker) = `1/sqrt(sum_k ||x_0||^{2(k-1)})`; `0.707` at `gamma = ||x_0||` (`K = 2`). The circuit cost is missing |
| block normalization | T-bare lift | section 2.5 table (`s x max`, LCU, `||G||`) |
| singular-value certification | T-bare, stacked backward Euler | `kappa` `12,071`, `166,189`, `11,558,518` at `T = 4, 20, 60` (ARPACK, with an upper bracket within `0.003` percent); no certified `sigma_min` |
| Newton or nonlinear feedback | none for a lift; for a linearization | no admissible number. The `0.17` / `1.07` pair used the exact trajectory as input (T-physical-input diagnostic only) |
| truncation | T-bare lift | the first-block error is the propagated tail (Duhamel); origin-centred observable errors in section 2.1; tube admissible defect `4.48e-5` / `1.07e-7` (section 2.3) |
| integration | T-bare lift | RK4 step error `<= 2.3e-6` (`K = 4`, `T = 20`); the Duhamel quadrature is Simpson to `<= 4.1e-9` |
| readout | per contract | postselection `w^2` (section 2.5); amplitude estimation `pi ||c|| ||x|| / epsilon` (section 2.6); union-bound factors 4.4 to 15.4; a sign reference for `x_0`; and, for the windowed route, a 31-coordinate readout per re-lift (19 to 79) |
| physical repair | T-repaired (a separate target) | 0.80 of the classical wall (draft block C); nothing in sections 2.1 to 2.8 applies to it |

A Carleman scalar product is not a physical Gram entry when the lift is
truncated (correction 11). An exact `(1, X_1, X_2)` does evaluate the
retained joint Gram linearly.

## 5. Requirement 4 (replaces L256-307)

- **Reductions: map and inverse.**
  - Centre mode: `S` and `S^{-1}` in `conserved_structure.centre_mode_basis`;
    round trip `0.0`.
  - `ell`: elimination and reinsertion in `ReducedForm.embed` and
    `ReducedForm.restrict`.
  - Both are exact linear reductions. Nonlinear level-set reductions by the
    two cubic integrals are not assessed (correction 15).
- **Neutral modes, projections, gap, sectors, nonresonance, common metric:**
  section 2.4. All fail on the reduced system.
- **Trajectory bound.** The largest `|x_i|` along T-bare is `1.50`, `1.68`,
  `2.11` to `t = 4, 20, 60` (a coherent `B` coordinate), with escape at
  `t = 141.4059`. Nothing bounds the off-diagonal `C` entries (correction
  15).
- **Drive treatment.** No common metric exists (correction 19). The driven
  propagator is priced directly:
  - for the linear part `A_1 + f D`: `12.086` (G2);
  - for the route-relevant `J(t)`: section 2.8.
- **KvN.**
  - Map and inverse: a packet, and its mean.
  - Inverse bias: section 2.7.
  - Register and cost: correction 21.

## 6. Implementation seams (replaces L341-367), with path:line

**Existing, committed at `a23ff642`, in
`paper_5/src/paper5/stability/quantum_target_form.py`.** The line numbers
are after this revision's docstring correction (+6 lines):

- extraction:
  - `extract_cubic_polynomial:96`
  - `ClosedPolynomialForm:192`
  - `extract_closed_polynomial_form:253`
- structure reports:
  - `block_coupling_table:385`
  - `tensor_statistics:427`
  - `linear_part_spectrum:481`
- the lift:
  - `carleman_lift:525`
  - `carleman_block_derivative:559`
  - `carleman_truncation_remainder_norms:615`
  - `carleman_sparse_generator:664`
  - `integrate_carleman_rk4:739`
- energy and linearization:
  - `closed_internal_energy:805`
  - `logarithmic_norm:821`
  - `integrate_linearized_on_samples:833`

**New in this revision (uncommitted).**

- **Seam C, `paper_5/src/paper5/stability/conserved_structure.py`:**
  - `centre_mode_basis:42`
  - `linear_first_integral:59`
  - `ReducedForm:72`
  - `reduce_at_centre_mode_fixed_point:118`
  - `neutral_mode_report:167`
  - `order_two_resonances:193`
  - `generated_lie_algebra:224`
  - `common_metric_obstruction:256`
  - `skew_adjoint_metric_dimension:277`
  - `signed_sector_report:291`
  - Tests: `paper_5/tests/test_conserved_structure.py`, 6 tests.
- **The parity checks:** `paper_5/tests/test_archive_hamiltonian_parity.py`,
  11 tests. The second transcription lives in the test file on purpose: it
  must not share code with `src`.

**Still proposed (no code).**

- **Seam A**, `quantum_output_contract.py`.
- **Seam B**, `lifted_propagation.py`: it would port
  `run_task3_tail.py`, `r2/c1_o3_precision.py` and `r2/c3_pair_lambda.py`.
- **Seam D**, `koopman_generator.py`: there is no prior code beyond
  `r2/c7_kvn_bias.py`.
- **Seam E**, `linearized_solver_ledger.py`: it would port the
  instability agent's `cond_stacked.py`.

**What is still script-only.** The constant and affine Poisson-tensor
searches, the first-integral enumeration and the commutator check exist only
in gitignored scripts:

- `run_task2_structure.py`, `run_task2_affine.py`, `run_task2_offdiag.py`;
- the refuters' `G1-structure/s*.py`.

## 7. Disagreements with the verification

None of the refuted claims is overturned.

One addition to G4 (correction 10). G4 found `H` read from block 2 at
`K <= 3` still above `1e-2` against T-bare (`0.029` to `0.18`). At `K = 4`
and `T = 4` it is `0.0094`, which is the only origin-centred lift reading
under `1e-2`; the site-0 occupation at the same order is `0.0915`. The
conclusion, that origin-centred lifts do not meet O3, stands.

## 8. Next bounded calculation

-> **Readout-error propagation for the re-centred windowed lift.**

- **Setup.** Add Gaussian readout noise `epsilon_r` per coordinate at each of
  the 19 (`K = 3`, window 1) or 79 (window 0.25) re-lifts.
- **Measure.** The site0 and `H` errors at `T = 4, 20` against T-bare, over 64
  noise draws, for `epsilon_r` in `{1e-4, 1e-3, 1e-2}`.
- **Output.** This gives the readout precision per re-lift that keeps O3
  within `1e-2`, which with section 2.6 prices the windowed route end to end.
- **Cost.** About 64 x 6 runs of under 8 s each; under 20 minutes split in
  two processes.

**Unresolved:**

- the user's decision (section 0);
- certified `sigma_min` and state-preparation circuits (none exist);
- nonlinear level-set reductions (not assessed);
- the order-4 adjoint at `T = 20` (not run);
- G2's KvN compression register (558 to 1,023 qubits): a heuristic is still
  needed for a sufficient cost.

## 9. Hashes of this revision's files

All uncommitted at writing. This file cannot carry its own hash; it is given
in the final report.

| file | SHA-256 |
|---|---|
| `MATH/paper_facing/paper_V_high_u_gkba/quantum_integration_feasibility_20261010.md` (with the appended corrections section) | `72d3414c425948351a29af06ce2b74e2b313519eee6d42f2cdd5f6827961a1a1` |
| `paper_5/src/paper5/stability/quantum_target_form.py` (docstring corrected only) | `8a647ff19fc01b1726c401d292b139b6665057e080b1c3ac3ec5cca68e374234` |
| `paper_5/src/paper5/stability/conserved_structure.py` (new) | `2188b0812bb5dfbafd79326bd0dea12a859c4300b595520d64a1127ce008393e` |
| `paper_5/tests/test_archive_hamiltonian_parity.py` (new, 11 tests) | `ad53e7e34794b50d9d707f5a25b6b8154dcad07a32d8cc2531aa5fbde7adb3ea` |
| `paper_5/tests/test_conserved_structure.py` (new, 6 tests) | `f47f4553f28f0e1dbfa6e86a62841e885c57797231c7a73a2f2f00582cb3eb46` |
| `paper_5/notes/electron_phonon_closure_worklog_20260803.md` (block appended) | `9d128db1a9debf118782ff095da86e0fb7e8d32b1cc5488d42b9789ef65d48f5` |
| `r2/c1_origin.json`, `c1_k4.json`, `c1_recentred.json` | `6d9899cb...a44f`, `1a78fa32...155e`, `9a86e6b8...970f` |
| `r2/c2_parity_receipt.json` | `07afc69e...ba24` |
| `r2/c3_pair_lambda.json` | `f510a0c0...4ac2` |
| `r2/c4_reduced_system.json` | `658a81f6...1168` |
| `r2/c5_ledger.json` | `a8f6d120...4210` |
| `r2/c6_sign_scale_confidence.json` | `607b040a...2ebe2` |
| `r2/c7_kvn_bias.json` | `3ee483ee...2cd0` |
| `r2/c8_driven_linearized_propagator.json` | `e95d273a...cf82` |

Tests: `PYTHONPATH=paper_5/src pytest -p no:cacheprovider` on
`test_quantum_target_form.py`, `test_archive_hamiltonian_parity.py` and
`test_conserved_structure.py` gives **35 passed** (18 + 11 + 6) in 15.5 s.

Files to edit (next increment): `output/local_runs/paper_v_quantum_target_form_20261010/r2/c9_readout_noise.py` (new), `MATH/paper_facing/paper_V_high_u_gkba/quantum_integration_feasibility_20261010.md` (append), this file (append).
