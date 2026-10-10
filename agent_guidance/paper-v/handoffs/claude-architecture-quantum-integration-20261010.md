# Claude architecture reply B-Q3: quantum integration of the archive equations

Date: 2026-10-10. This is the persistent Paper V quantum-integration agent
(Claude), answering coordinator message 005 and the Codex B-Q2 theory dossier.

Status:

- This file holds the architecture decisions, the implementation seams and
  one bounded calculation, which was run.
- Exploratory, not promoted.
- Nothing committed by this agent.
- No manuscript edit.
- No edit to `matrix_reference.py` or `hubbard_dimer.py`.

Findings that are not architecture go in
`MATH/paper_facing/paper_V_high_u_gkba/quantum_integration_feasibility_20261010.md`
(Tasks 1 and 2). This file cites them.

| Anchor | Value |
|---|---|
| source | worktree `Holstein_test.worktrees/claude-paper-v-routes-20260914`, branch `claude/paper-v-routes-20260914`, HEAD `6b27596e` |
| Python | `/Users/lordskyler/Holstein_test/.venv/bin/python`, `PYTHONPATH=paper_5/src` |
| run directory | `output/local_runs/paper_v_quantum_target_form_20261010/` (gitignored) |
| Codex dossier read | `quantum-ode-theory-dossier-20261010.md` on `codex/paper-v-quantum-ode-theory-20261010`: requested commit `92ebd4ab` (blob SHA-256 `c93a4de8...85ad`); the file read is its successor `f96678da` (SHA-256 `817e696a...b8c3`), which only adds the final instability-report reconciliation |

---

## 0. Decisions and negative results (read first)

1. **The propagated omitted tail is now measured exactly.** It equals the
   recovered-coordinate error to integration accuracy: the identity holds to
   `2.6e-6` to `5.2e-6` relative.
   - The norm certificate that bounds that error is loose by `22` to `657`
     times.
   - The terminal first-block weight falls with the order: `0.26`, `0.056`,
     `0.0078` at `T = 4` for `K = 2, 3, 4`; `0.18`, `0.032`, `0.0045` at
     `T = 20`.
   - The pointwise tail ratio of Task 1 (median `0.72`) predicts neither
     number. Section 3.
2. **No conserved-system reduction with a usable structure exists in the
   classes tested.**
   - No constant or affine Poisson tensor generates the drive-free field,
     for any Hamiltonian.
   - The only exact reductions are the decoupled centre-mode coherent field
     (2 coordinates, solved classically) and the one linear first integral
     (1 coordinate).
   - Four of the five neutral zero modes of `A_1` receive a nonzero
     quadratic projection. So Jennings Theorem 4.1 fails after any exact
     reduction available here, not only before it.
   - There is no positive definite metric common to `A_1` and `D`. The
     driven linear propagator is priced directly: supremum `12.09`
     (Euclidean), `7.23` in the `A_1` metric, to `t = 20`. Section 4.
3. **Koopman--von Neumann needs no Hamiltonian structure but is dominated.**
   It needs about 430 to 680 logical qubits and `T ||L|| ~ 5e5` to `6e9` for
   one trajectory at `1e-2`. Direct simulation of the exact model needs 12 or
   19 qubits and `T ||H|| ~ 510` to `680`, with no closure error. Task 2(b) of
   the report.
4. **The output contract is unresolved.** The draft supports four distinct
   contracts, listed in section 1. **No route can be selected before the user
   picks one.** This is the one decision needed. Section 1 ends with a
   recommendation.

---

## 1. The output contract (requirement 1)

**What the draft asks for, verbatim anchors in
`MATH/paper_details/paper_V_high_u_gkba.tex`:**

- McLachlan pathway, L=2 benchmark: "compare the reconstructed trajectory
  against Radau/BDF/DOP853 classical references". Its diagnostics are "the ODE
  residual, sector residuals, trajectory errors for x and B, Tr rho,
  Hermiticity, positivity drift, and growth of" the correlation.
- NISQ layer: "residual and invariant estimation for selected observables,
  with full trajectory reconstruction left as a small-system diagnostic".
- Fault-tolerant linearization layer: a credible claim "must specify ... the
  observable extracted from the amplitude-encoded output".
- Decision rule item 4: no full quantum solution may be claimed "unless
  Eq. (B), nonlinear-product access, and output extraction are all included in
  the cost model".

**What the record actually scores** (block A of
`numerical_plan_draft_20260920.tex`, the score scripts):

- the all-coordinate scalar RMS of the 31 closed coordinates against the
  exact cutoff-16 reference, over samples `t <= T` at `T = 4, 8, 12, 16, 20`;
- the Eq. (22) energy RMS;
- the physicality minima (electronic eigenvalue, retained joint Gram,
  positive extension).

The four candidate contracts, kept distinct:

| contract | object returned | norm and precision | sign and scale recovery | horizons and sampling | readout cost driver |
|---|---|---|---|---|---|
| **O1 full time series** | all 31 coordinates on the `0.0025` grid (8,001 samples to `t = 20`) | all-coordinate scalar RMS, absolute | required for every coordinate: amplitude encoding returns `z = x/||x||` up to a global phase, so `||x||` must be estimated separately and signs fixed against a reference | `t = 0..20` (and to `60` for the repair horizon); escape at `t = 141.41` from the exact correlated start | `31 x N_t x O(W/epsilon)` amplitude estimations; history-state solvers give the time series only with postselection on the time register |
| **O2 terminal state** | all 31 coordinates at `t = 4` and `20` | Euclidean or scalar RMS at `T` | as O1 | `T = 4, 20` | `31 x 2 x O(W/epsilon)` plus terminal first-block weight (section 3) |
| **O3 selected observables** | linear functionals `c^T x` (site-0 occupation `(1 + x_0)/2`; phonon occupation `N00 + N11`; Eq. (22) in its linear part) and quadratic ones `x^T Q x` (Eq. (22) energy: block 2 of a lift supplies `x (x) x` directly) | absolute error per observable | the overlap `<c, x>` needs a reference state with known sign, and the scale `||x||` | per observable and time | `O(1/epsilon)` per observable and time; the observable's covector times `Phi(T, s)` can suppress or expose the unstable mode |
| **O4 invariants and residuals only** | `Tr rho`, Hermiticity, Eq. (22) drift, residual norms | absolute | as O3 | instantaneous | lowest; not a trajectory claim |

The precision scale that matters is set by the target (section 5):

- **Physical reference.** The bare closure is `0.111`, `0.193` and `0.325`
  scalar RMS from exact at `t = 4`, `20` and `60`. Integration error below
  about `1e-2` is invisible behind that closure error.
- **Bare closure.** The record integrates at DOP853 `rtol 1e-12`.

Confidence is `1 - delta` per amplitude estimation. A union bound over
`31 N_t` estimates sets `delta` per estimate to `delta_total / (31 N_t)`.

**Recommendation -> O3 at `T = 4` and `20`.** Use the site-0 occupation and
the Eq. (22) energy, against the physical reference, at `epsilon = 1e-2`
absolute. It is the only contract whose readout does not scale with
`31 N_t` and whose precision is not already exceeded by the closure's own
error. O1 is a classical-diagnostic contract and should stay classical. This
is the user's decision.

## 2. Source locks and the independent parity tests (requirement 2)

| artifact | lock |
|---|---|
| instability report `agent_guidance/paper-v/overnight/closure-instability-20261010.md` | commit `40771435`, SHA-256 `751cab34895ee314c2babbea837f0924c2de0f77830a10574a56748658a5ae71` |
| its `summary.json` | `c606bee027bdde7f13d47a6654a2623a5d84d02fdd5a11b7ec5d58ba5d0fa16f` |
| its `verify_batched.json` | `1fdcdae9d2f59e00f99511854ab9a6d78154d435361c34e643b273fa1adbfbfc` |
| its `cond_exact_correlated_t0_t0_0.json` (stacked conditioning, exact correlated start) | `d3485acc372e0954647f60b166d965ba656d833165009f3c4e1c6dffad4579eb` |
| its `tangent_exact_correlated_t0_t0_0.json` (amplification, same start) | `54e187a89d8fb895f974e1d76e40e461e5f0e3cd611d1191031952670881fcc7` |
| its `cond_exact_t60_t0_60.json`, `tangent_exact_t60_t0_60.json` | `2e60130358543d809a1a2ba0f1fabf1751519840107554a2cdcb9bd314588aba`, `4539d8afe6aacf4c9a343013f7e1fb6b263a0fe552d706a04e080b28a8107a69` |
| its `cond_hartree_fock_t0_t0_0.json`, `tangent_hartree_fock_t0_t0_0.json` | `5c4777fea4cb4dcd8198fa66d06ff1a5e359c441ae7afd44515b5b81c312395f`, `4cdfe5650fcae9abb9bd55cc5a29ea68c09da804896194ab19ce7f01656cd8f2` |
| its `cond_repaired_chain_t0_0.json`, `tangent_repaired_chain_t0_0.json` | `2baadc67121e903649fd155e0bac9337ad570647874ae838c09890c41480646b`, `2feb9f3e07ac073d45b9dda0552582481c263f9a1f013ab8a33b2d7ccf6f2d27` |
| its `cond_exact_reference_t0_0.json`, `tangent_exact_reference_t0_0.json` | `13bb654e72552b5e216be582aad25994bf73ccb3a2509d3cdc2c95539101d0e5`, `7dfbbc1827f49c61b7ef24308cf9783d3ea0f44de2199163c413e32944024be1` |
| `matrix_reference.py`, `hubbard_dimer.py` (unchanged since the dossier) | `15db069e...3e329006`, `c2e5bd20...49d23087` |
| `quantum_target_form.py` | committed `d110f2dd` (`5e069335...137716d9`); live file with Task 2 and 3 additions (uncommitted) `1d8b520c...cd372b8c3` |
| extracted tensors `task1_form.npz`, `task1_form.json` | `48660577...7d05d350`, `56f0c5b5...a36c43` |
| manuscript `MATH/paper_details/paper_V_high_u_gkba.tex` | `00afb874...c660d` (unchanged) |

**Cubic wording, reconciled.**

- The committed instability report (`751cab34`, line 114) already says the
  field "is a polynomial of degree two --- the cubic parts of the
  normal-particle and normal-hole sources cancel". The dossier's flag applied
  to an earlier, uncommitted hash.
- The **manuscript** still carries a cubic term: `C(x,x,x)` in
  Eq. (`eq:paper_v_force_polynomial`) and `A_3 x^{(x)3}` in the lifting
  layer. That correction is the user's decision, recorded in the feasibility
  report under Task 2.
- The module keeps a cubic slot (`A_3`) as a generic interface. For this
  target it is asserted zero by a test.

**Independent verifications.** Each is a test; "exists" means it is in
`paper_5/tests/test_quantum_target_form.py` (18 passed); "specified" means it
is still to be written.

| property | test | status |
|---|---|---|
| coefficient parity, independent derivation | rebuild `F_0`, `A_1`, `D`, `A_2` **by hand-coded index contraction of Eqs. (14a)--(14e)** in closed coordinates. This is a second implementation that does not call `closed_scalar_rhs`. Compare entrywise to `1e-14`. Also compare against the instability agent's batched field (`verify_batched.json`: `6.1e-14` relative at scale 500) | specified: `test_independent_coefficient_derivation` |
| right-hand-side parity | `form.evaluate` against `closed_scalar_rhs` at random states and times (`test_polynomial_form_reproduces_committed_rhs`). Extend to the stored states of the exact reference, the bare trajectory from the exact correlated start, the Hartree--Fock trajectory and the repaired chain, at `1e-13` relative | exists; extension specified |
| exact interpolation on a known cubic | `test_cubic_interpolation_recovers_a_synthetic_cubic_exactly` | exists |
| the forcing | `F_0` has exactly five nonzeros, with closed-form values: `-g` on `coherent_q_imag` (from `-i 2 g rho_qq` at `rho = I/2`) and `-+g/4` on three correlation imaginary parts (bare Pauli source `-i (1 - rho) g rho` at `rho = I/2`) | specified: `test_forcing_closed_form` |
| the multiplicative drive | `F(t,x) - F(0,x) = f(t) D x` at random `(t, x)`, plus degree-0/2/3 drive parts zero at two probe times (`test_drive_enters_only_through_a_skew_linear_term`). **Add:** `D x` equals the closed coordinates of `-i [sigma_z/2, rho]` and `-i [sigma_z/2, C_q]` computed from the matrix fields | exists; analytic form specified |
| trace-zero identity | `tr A_1 = tr D = 0`, `sum_i A_2[i,i,:] = 0` (`test_archive_rhs_has_no_cubic_part_and_is_divergence_free`). **Add an independent route:** central-difference divergence of `closed_scalar_rhs` itself at 100 random states with step `1e-4` (error `O(1e-9)`), not via the tensors | exists; independent route specified |
| energy identity and drive power | `test_internal_energy_is_quadratic_and_conserved_identically` | exists |
| sparse lift parity | `test_sparse_lifted_generator_matches_the_block_velocity` (`K = 1, 2, 3`, synthetic cubic with drive) | exists |

## 3. The propagated omitted tail and the terminal first-block weight (requirement 3)

This was the first bounded calculation and it was run: `run_task3_tail.py`,
`task3_tail_2.json` (`01734edc...`) and `task3_tail_3.json` (`3bb278fb...`);
`K = 2` took 9 s and `K = 3` took 277 s.

**Definitions.**

- The target is the bare closure's own trajectory `x(s)` from the exact
  correlated preparation.
- `X~` is the order-`K` truncated lift solution from `lift(x(0))`.
- With matched lift initialization and matched forcing, the error obeys this
  exact identity:

  ```text
  P_1 (lift(x)(T) - X~(T)) = int_0^T P_1 U_K(T, s) tau_K(s) ds
  ```

  where `U_K` is the lifted truncated propagator. Because `A_3 = 0`,
  `tau_K(s)` lives in block `K` and equals
  `sum_m x^{(m)} (x) A_2(x, x) (x) x^{(K-1-m)}`.
- `W(s) = P_1 U_K(T, s)` is computed by the adjoint equation, backward RK4 at
  `h = 0.005`, with `tau_K` at the stored closure samples.
- Terminal first-block weight is `||X~_1(T)|| / ||X~(T)||`.

| `K` | `T` | measured `||x(T) - X~_1(T)||` | identity mismatch | certificate `int ||P_1 U_K(T,s) P_K|| ||tau_K(s)|| ds` | certificate / measured | `sup_s ||P_1 U_K(T,s) P_K||` | terminal first-block weight | initial first-block weight |
|---|---|---|---|---|---|---|---|---|
| 2 | 4 | 2.246 | `2.6e-6` | 50.2 | 22.4 | 17.1 | 0.261 | 0.401 |
| 2 | 20 | 19.65 | `3.4e-6` | 2,886 | 147 | 142.7 | 0.182 | 0.401 |
| 3 | 4 | 2.045 | `3.9e-6` | 222.1 | 109 | 37.7 | 0.056 | 0.173 |
| 3 | 20 | 127.0 | `5.2e-6` | 83,506 | 657 | 2,746 | 0.032 | 0.173 |

Two further measurements:

- **Order 4** (from the stored runs; the adjoint was not run at
  `954,304` dimensions): terminal weight `0.0078` (`T = 4`) and `0.0045`
  (`T = 20`), initial `0.076`.
- **Norm ratio.** For `K >= 2`, `max_t ||X~(t)|| / ||X~(T)||` is `1.0`, or
  `1.04` for `K = 2` at `T = 20`: the lifted norm grows monotonically, so the
  norm-ratio factor `g` of Berry et al. and Krovi is benign. The cost sits in
  the terminal weight and in the propagated tail.
- **Step refinement at order 4 to `T = 20`**, requested by the dossier:
  `h = 0.005` against `0.0025` differs by `2.3e-6`, against a first-block
  error of `387` (`check_steps_k4_t20.json`).

**Reading.**

- **Certificate.** The propagated-tail integral reproduces the measured
  error, so the error is the propagated tail and nothing else. The norm
  certificate built from the same two factors is too loose to certify
  anything at `epsilon = 1e-2`: it is 22 to 657 times the actual error, which
  is already of order one.
- **Extraction.** Extracting the first block from an amplitude-encoded lift
  succeeds with probability equal to the terminal weight squared: `0.068`,
  `3.1e-3` and `6.0e-5` at `T = 4` for `K = 2, 3, 4`. Amplitude amplification
  needs a factor `1/weight` more rounds: 3.8, 18, 129.
- Both factors worsen with `K`, while `K` buys accuracy only for `t <= 2`.

**Continuous amplification and nonlinear tube, planned separately from these
sampled diagnostics.** The calculation above samples `s` at `0.005`. To
certify it:

- **(a) Continuous envelope of the lifted propagator.** Between grid points,
  `||U_K(t', t)|| <= exp(h mu(A_K))`, with `mu` the Euclidean logarithmic norm
  of the lifted matrix. Its bound is cheap from the sparse generator, as the
  largest eigenvalue of the symmetric part via `eigsh`. This gives a
  certified supremum at most `exp(0.005 mu)` above the grid maximum.
- **(b) Continuity of the tail.** Bound `||tau_K(s') - tau_K(s)||` by
  `|s' - s|` times a derivative bound from `max ||F||` and `||A_2||` on the
  tube.
- **(c) Nonlinear tube for any approximate integration of the bare closure.**
  The bound is

  ```text
  r(T) <= int ||Phi(T,s)|| (delta(s) + ||A_2|| r(s)^2) ds
  ```

  - **Tube condition.** With `Lambda = sup ||Phi||` (instability report:
    `13.0`, `51.0`, `3,169` at `T = 4`, `20`, `60` from the exact correlated
    start), write `a = Lambda T delta_avg` and `b = Lambda T ||A_2||`. The
    fixed-point inequality `R <= a + b R^2` has a solution whenever
    `4 a b <= 1`, that is `4 ||A_2|| Lambda^2 T^2 delta_avg <= 1`, and then
    `R <= 2 a`.
  - **Resulting `delta_avg`.** With `||A_2|| = 1.995`, the defect per unit
    time must be below `4.6e-5` (`T = 4`), `1.2e-7` (`T = 20`) and `3.5e-12`
    (`T = 60`).
  - **Certification.** `Lambda` must itself be certified by (a) applied to
    the 31-dimensional `Phi`. The instability report's start grid of `0.5`
    is too coarse: `exp(0.5 x 2.45) = 3.4`. The adjoint here at `0.005`
    gives `exp(0.005 x 2.45) = 1.012`.

The sampled numbers stay labelled as sampled until (a) to (c) exist.

## 4. Conserved-system reductions and embeddings (requirement 4)

**Task 2's Hamiltonian and Poisson structure: none in the tested classes.**
Report, Task 2(a):

- No constant Poisson tensor generates `F`, with any Hamiltonian. The full
  field preserves one constant tensor, of rank 2.
- No affine one does either. The preserved affine family has dimension 4, all
  of rank 2 on the `B` and `N` coordinates.
- The minimal state-dependent tensor fails the Jacobi identity.

There is therefore no Darboux or Lie-Poisson map to specify.

**What exists exactly, with map, inverse and tests:**

| reduction | map | inverse | exactness test | what it removes |
|---|---|---|---|---|
| centre-mode coherent field | `B_+ = B_0 + B_1`, `B_- = B_0 - B_1` (real and imaginary parts) | `B_0 = (B_+ + B_-)/2`, `B_1 = (B_+ - B_-)/2` | `d B_+/dt = -i omega B_+ - 2 i g` exactly, because `rho_00 + rho_11 = 1`. Test: the rows of `F_0`, `A_1`, `D`, `A_2` for `B_+` involve only `B_+`. The preserved rank-2 tensor and the first integral `|B_+ + 2g/omega|^2` (drift `2.5e-17`) confirm it | 2 real coordinates; a closed-form classical oscillator |
| the linear first integral `ell` | eliminate `correlation_shared_trace_real` by `s_r = ell_0 - (g/6)(N00 + N11 + 2 Re N01)` | reinsert | `ell^T [F_0, A_1, D, A_2] = 0` (singular value `1.8e-15`) | 1 coordinate |

After both reductions, 28 coordinates remain. The dossier's conditions,
measured or specified for that reduced system:

| condition (Jennings v2 and the dossier) | status on the reduced system |
|---|---|
| neutral left modes annihilate the nonlinearity and the forcing | **fails.** `A_1` has 5 zero modes. All 5 left vectors annihilate `F_0` (rank of `[F_0, A_1]` is 26 = rank `A_1`), but only one of them annihilates `A_2` (rank of `[F_0, A_1, A_2]` is 30). So 4 zero modes receive a nonzero quadratic projection. The imaginary modes, all neutral, also receive it, as the dossier argues. Test to write: the projections `v_i^T A_2 (x (x) x)` per left eigenvector, as coefficient norms |
| a strictly stable complement with gap `delta > 0` | **fails structurally.** `tr J = 0` survives the linear reductions (the trace of the reduced Jacobian is the trace of the full one minus the removed block's, which is zero for the oscillator and for the eliminated integral) |
| signed sectors S+ and S- (Theorem 5.12) | **fails.** The zero modes fit neither sector |
| nonresonance | **fails automatically** at order three for conjugate imaginary pairs (`lambda_k = lambda_k + i w - i w`), as the dossier states |
| trajectory bound | finite-horizon only: maximum absolute coordinate `1.50`, `1.68`, `2.11` to `t = 4`, `20`, `60`; escape at `141.41`. No all-time bound exists |
| drive treatment | **no common metric.** On the 8-dimensional space of symmetric `G` with `A_1` and `D` both skew-adjoint, the largest attainable smallest eigenvalue at trace 31 is `1.9e-8` (SCS), i.e. none is positive definite. The task-1 metric fails `D` at relative `0.21`. **Priced directly:** the driven linear propagator `A_1 + f(t) D` has `sup ||Phi(t,s)||` of `12.09` (Euclidean; undriven `12.08`) and `7.23` in the task-1 metric (undriven `1`), to `t = 20` (`task2_metric.json`) |

**Koopman--von Neumann.**

- **Map.** `x` goes to a normalized packet `psi_0(x)`, a Gaussian of width
  `sigma_0` centred at `x(0)`, on a box of width `W ~ 4` to `5` per
  coordinate. The generator is `L(t) = -i F(t, x) . grad`, since
  `div F = 0`; symmetric ordering makes the discretized generator exactly
  skew-Hermitian.
- **Inverse (readout).** The mean `<x_i>_psi(T)` by amplitude estimation.
  This is not the characteristic: `E F(X) != F(E X)`, with a bias of order
  `||A_2|| sigma(T)^2`.
- **Tests to specify:**
  - unit tests of the discretized generator on the 2-coordinate centre-mode
    oscillator, which is exact and linear;
  - a 3- or 4-coordinate frozen-coefficient sub-block, classically on a grid;
  - packet-mean tracking error against `sigma_0` at `T = 4`;
  - boundary flux out of the box before the escape time.
- **Drive.** The drive is a time-dependent generator (Dyson-series or
  interaction-picture Hamiltonian simulation), with no extra condition. Under
  the working split, `B(t)` is a prescribed coefficient.
- **Cost.** In the report, Task 2(b).

## 5. Architecture and cost ledger (requirement 5)

Four targets, never substituted for one another:

| target | definition | polynomial ODE? | what the lift or KvN approximates |
|---|---|---|---|
| **T-bare** | `closed_scalar_rhs` trajectory, 31 coordinates including `B` | yes, quadratic | this |
| **T-repaired** | cheap-repair chain: fixed dictionary, Dykstra projections, re-selection | no: nonsmooth projections and selections | nothing in sections 3 and 4 applies; its derivative needs the repair maps |
| **T-fixed-B** | Eqs. (14a)--(14d) on 27 coordinates with `B(t)` classical | yes, quadratic in 27 coordinates, with `A_1(B(t), t)` time-dependent; quadratic part only `(rho)(rho)`, `(rho)(N)`, `(rho)(A)` | a different lift: no `(rho)(B)` or `(B)(C)` quadratic terms; adds `B`-feedback error |
| **T-physical** | exact cutoff-16 moments | no closure | the closure's own error (`0.111`, `0.193`, `0.325` RMS at `t = 4`, `20`, `60`) is a floor against it |

**A Carleman scalar product is not a physical Gram entry.**

- Block 2 of a lift holds `x_i x_j`, products of retained expectation values.
- The repair's retained joint Gram and positive extension hold
  `<O_i^+ O_j>` over a dictionary that includes higher moments absent from
  the 31 coordinates.
- No linear map from `X_K` evaluates the repair's cones.

Ledger, one line per cost, never merged:

| line | what it is | measured or known now | certificate still missing |
|---|---|---|---|
| preparation | initial lift `lift(x(0))` (Kronecker powers of a 31-vector) or KvN packet; for T-fixed-B also `B(t)` access | first-block weight at preparation `0.40`, `0.17`, `0.076` (`K = 2, 3, 4`) | a state-preparation circuit and its cost for a product-state-like lift |
| block normalization | `alpha_BE` of the sparse lifted generator (`L`, `M`, `b` from `carleman_sparse_generator`); for linearization, of `I - h J` | nonzeros of `L`: in `task3_tail_*.json` (`K = 2, 3`) | `alpha_BE` for the chosen access model (sparse-access `s max|entry|`, or an LCU of the Kronecker terms) |
| singular-value certification | `sigma_min` of the lifted or stacked system | stacked backward Euler `kappa`: `1.1e4` to `1.16e7` (instability report), against `2,038` to `30,559` from the step count alone | certified `sigma_min` (ARPACK estimates only so far) |
| Newton or nonlinear feedback | none for a lift; re-linearization for the linearization layer; `B` update for T-fixed-B | the linearization about exact tracks the closure to `0.17` at `T = 4` and fails by `T = 20` (`1.07`) | Newton basin, iteration count, feedback error |
| truncation (lift) | propagated tail, section 3 | exact identity measured; certificate loose by 22 to 657 times | continuous envelope (a) and (b) |
| integration | RK4 or the solver's own | `<= 2.3e-6` against truncation errors `>= 1e-3` | -- |
| readout | first-block postselection then amplitude estimation per observable; sign and scale | postselection probability `weight^2`: `0.068`, `3.1e-3`, `6.0e-5` at `T = 4` | sign-reference construction; norm estimation for `x = r z` |
| physical repair | Gram projections, re-selection | 0.80 of the classical wall (draft block C) | the dossier's matrix-function ledger; not addressed by any route here |

## 6. Implementation seams in `paper_5/src/paper5/stability/`

`quantum_target_form.py` is the base. It is committed at `d110f2dd` and
extended (uncommitted) in this round. It provides:

- exact extraction: `extract_cubic_polynomial`, `extract_closed_polynomial_form`;
- structure reports: `block_coupling_table`, `tensor_statistics`,
  `linear_part_spectrum`;
- the lift: `carleman_lift`, `carleman_block_derivative`,
  `carleman_sparse_generator` (new), `carleman_truncation_remainder_norms`,
  `integrate_carleman_rk4`;
- `closed_internal_energy` (new), `logarithmic_norm` (new) and
  `integrate_linearized_on_samples` (new).

Proposed seams, each a new module with its own test file:

| seam | module | responsibility | first test |
|---|---|---|---|
| A | `quantum_output_contract.py` | a frozen `OutputContract`: target, observables as linear `c` or quadratic `Q` functionals, times, norm, `epsilon`, `delta`, sign and scale recovery. Pure function `error_budget(contract, amplification, ...)` splitting `epsilon` into model, truncation, integration, preparation, feedback and readout lines | allocations sum to `epsilon`; an O3 contract with `c` orthogonal to the leading left singular vector gets a larger truncation share |
| B | `lifted_propagation.py` | `propagated_tail(form, order, trajectory, horizon)`: adjoint `W(s)`, identity check, certificate, terminal weight. `lifted_lognorm(form, order)` for envelope (a). Move the logic of `run_task3_tail.py` here | identity mismatch below `1e-5` on the `K = 2` case; on the Riccati scalar, the tail integral equals the Taylor remainder |
| C | `conserved_structure.py` | polynomial first integrals (sampled nullspace); preserved constant and affine bivectors; the centre-mode and `ell` reductions with map and inverse; neutral-mode projections of `F_0` and `A_2`; common-metric SDP | reductions round-trip to `1e-15`; the reduced field's rows match the full field's; the projection counts reproduce `4 of 5` |
| D | `koopman_generator.py` | symmetric-ordered KvN generator on a small grid for sub-blocks; resource estimator (qubits, `T ||L||`, readout) from `(Lambda, W, epsilon, sum max|F|)` | skew-Hermitian to rounding; centre-mode oscillator packet tracks the closed-form solution |
| E | `linearized_solver_ledger.py` | stacked backward-Euler builder, adopted from the instability agent's `cond_stacked.py`; certified `sigma_min` by shift-invert with a residual bound; `alpha_BE` per access model | `J = 0` reproduces `sin((2m-1)a)/sin a`; the bracket check agrees with ARPACK |

Rules for all seams:

- Never import an exact reference into a propagation path.
- Exact references enter only as reporting inputs.
- No default changes; no edits to `matrix_reference.py` or `hubbard_dimer.py`.

## 7. Next bounded calculation (under 20 minutes)

-> **Recommended: certify the closure's amplification envelope and the
nonlinear tube, section 3, items (a) and (c).**

- Use the adjoint at `0.005` on the 31-dimensional `Phi` along the bare
  trajectory from the exact correlated preparation, for `T = 4, 20`.
- Bound each grid interval by `exp(0.005 mu_2(J))`.
- Report the certified `Lambda(T)` and the admissible defect `delta_avg` for
  a tube.

It costs about 1 minute; there is no lift. It converts the instability
report's sampled supremum into the certificate every route's error budget
needs.

The alternative is seam C's neutral-mode projection test, a few seconds,
which fixes the Jennings 4.1 failure as a test.

**Unresolved problems:**

- **The output contract (section 1)** is the user's decision.
- **The manuscript's cubic term** is the user's decision.
- **The propagated-tail certificate** is not yet continuous, and its looseness
  (22 to 657 times) means a certificate route needs the observable-specific
  form `c^T P_1 U_K` rather than norms.
- **Order 4** was not run through the adjoint (`954,304`-dimensional;
  feasible but over the time box).

Files to edit (next increment): `paper_5/src/paper5/stability/lifted_propagation.py` (new), `paper_5/tests/test_lifted_propagation.py` (new), `MATH/paper_facing/paper_V_high_u_gkba/quantum_integration_feasibility_20261010.md` (append).
