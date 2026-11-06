# Disposition of root reviews 027 and 030 (quantum part)

Date: 2026-10-18. Author: the persistent Paper V quantum-integration agent
(Claude), the existing owner of B-Q3 r2, the route (a) pricing and the
quantum consolidation. Requested by message `codex-to-claude-20261010-013`.

**Status.**

- Exploratory, not promoted.
- No route, target, output, default or implementation is selected.
- No manuscript edit; no state-changing git command.
- No new measurement. The new numbers below are arithmetic on stored
  receipts (`r4` in the receipts table at the end).

**Inputs, with SHA-256 recomputed before reading:**

- review 027, `3e6ecfc005219c5b21492a847680816f867c2ed56681b20ddc10ec9ba3784903`;
- review 030, `14c14fd4da0355840c944263d15cac0a6fb90fa6f683dad9fc533338f9442f3b`;
- r2 at `475c47a7`, `e8c88356...583bc`;
- pricing at `d3c80e8a`, `e98c046f...ddee3`;
- consolidation at `424adca9`, `8534130a...f501`.

**Where the amendments live.**

| file | treatment |
|---|---|
| pricing report | amended in place, with a dated revision note (35 statements) |
| consolidation | amended in place, with a dated revision note (52 statements) |
| r2 | not edited: it is a committed revision of the a23ff642 reply, and this disposition is its amendment record. Its statements reach the consolidation, where they are amended |

Committed versions remain in history. Line references with no file prefix
are to the committed versions.

**Count.** 15 rows: seven findings of 027, its timing section, and seven
quantum findings of 030. Of these, 12 are accepted and 2 accepted with
amendment (027-1, 027-5). The last, 030-6, is accepted except one detail,
survivor conditioning of the packet-width thresholds. That detail was
contested with the line relied on, and root has since resolved it as a
supported amendment (text L). None is contested in substance.

---

## The rows

| row | finding | disposition | lines it changes (committed version) | replacement text |
|---|---|---|---|---|
| 027-1 | The postselected first block encodes `Y_1`, so the recovered norm is `||Y_1||`, not `gamma = ||x_r||`; the 1.1e9 / 1.5e10 / 230 figures are not end-to-end costs | **accept with amendment** | pricing 19-23, 119-121, 181-183, 186-196, 236, 239-240; consolidation 610-612, 621-627, 643-645, 672-675; driver `r3_route_a.py:269` (not edited; receipts stand) | A |
| 027-2 | 19 restart batches plus two independent output batches make 21, not 20; a Gaussian `sigma` and four marginal p95 are not an amplitude-estimation tolerance or simultaneous confidence; the shots branch is a Gaussian surrogate with the true norm; `R = 19` is a finite comparison | **accept** | pricing 19-21, 51-60, 74-75, 108-122, 133-138, 145-152, 250-254; consolidation 591-612, 707-710, 807-808; r2:466 | B |
| 027-3 | Bind the time-dependent forcing, propagator and access costs: frozen `beta`, frozen sparse normalization, `sup||exp(G_fmax t)||` is not the time-ordered bound; preparation, discretization and readout unresolved; 0.19 s is one benchmark | **accept** | pricing 174-178, 190-196, 198-211, 247-249; consolidation 613-627, 632-637, 797-800 | C |
| 027-4 | Route (b)'s 4.9e6 omits normalization, preparation, simulation error, access and failure probability; the whole 1e-2 is assigned to readout; cutoff 16 against 20 is not an infinite-cutoff bound; the triangle inequality gives a configuration-specific limitation only | **accept** | pricing 213-240; consolidation 638-649, 659-664 | D |
| 027-5 | The pair-of-times tube is conditional; 13.22 / 54.10 are conditional estimates; the defect threshold is a conditional scalar criterion; no verified defect, restart-jump or readout budget for route (a) | **accept with amendment** (the helper evidence of the review file, section 5) | r2:251-253 (inequality written reversed), r2:261, r2:464-465; consolidation 463-470, 777, 811-813 | E |
| 027-6 | The common-metric witness is numerical (`||X^2||` 2.4e-13 / 4.4e-13, finite-time exponentials); numerical ranks and clustering need tolerance scope; fixing the centre mode is exact on an invariant submanifold, not for an oscillating preparation | **accept** | r2:107, 289, 300, 488; consolidation 317-324, 763; the centre-mode mismatch is already in r2:273-277 and consolidation 325-328 (kept) | F |
| 027-7 | The KvN `T ||L||` lower bound is a coefficient-wise proxy, not an operator-norm bound; mesh and register rules are heuristic; the `sigma_0 = 0.1`, `T = 20` bias is survivor-conditioned (916 of 1,000 pairs) | **accept** | r2:109, 395-398; consolidation 512-527, 757 | G |
| 027-T | Per-process timers do not establish the whole-increment bound | **accept** | r2:122; pricing 267-268; consolidation section 7 (added) | H |
| 030-1 | Scope the negative conclusion to the assessed routes (consolidation 655-687, "any lift must be re-centred" at 497) | **accept** | consolidation 497-498, 655-657, 661-664; r2:59-63 ("Only one route attains it") and r2:449 | I |
| 030-2 | Correct the readout ledger: `||Y_1||`, 21 batches, a surrogate rather than a verified estimator, marginal p95 rather than simultaneous confidence; 1.1e9 / 1.5e10 / 4.9e6 / 230 are incomplete proxies | **accept** (texts A and B) | consolidation 599-627, 641-645 | A, B |
| 030-3 | Keep the driven-family and implementation premises: `beta_j`, `S_j`, `C_max`, access, preparation, encoding, precision and confidence; query proxies against 0.19 s do not settle runtime advantage | **accept** (text C) | consolidation 613-637, 669-675 | C |
| 030-4 | Separate the closure discrepancy from total physical accuracy: the error lies between `max(0, Delta - epsilon)` and `Delta + epsilon`; direct simulation has no certified total allocation; the cutoff difference is not an infinite-cutoff bound | **accept** | consolidation 574-583, 628-629, 646-649, 660-664, 771; pricing 233-234 | D, J |
| 030-5 | Distinguish conditional algebra from floating witnesses: the nilpotent witness; numerical ranks; trace zero alone does not exclude a strictly stable complement; `J_min` needs `grad H != 0` and `F . grad H = 0`, and does not give Jacobi | **accept** (the gradient never vanishes here, as 030 notes) | consolidation 242-243, 289-291, 317-324, 371, 763; r2:297 | F, K |
| 030-6 | Preserve tube, operator and packet qualifications: inflated sampled `Lambda` is not a tube; coefficient-wise maxima are not operator-norm lower bounds; packet-width thresholds are protocol-specific extrapolations "with survivor conditioning" | **accept**, with the survivor-conditioning detail **contested** for the two thresholds | consolidation 463-470, 505-506, 512-527; r2:395-398 | E, G, L |
| 030-7 | Preserve 89 / 187 and the 137 s prediction label; a polynomial count is not a polynomial integration cost; state the encoding assumptions; full-coordinate readout is not the classical cost times overhead; 18 GB and 87 rows are historical, not a current cap | **accept** | consolidation 701-704, 712-714, 732-735 (89 / 187 at 693-697 and 137 s at 718-723 unchanged) | M |

---

## Replacement texts

The applied wording is in the amended files, located by their revision
notes. The texts below are the substance.

**A. The encoded increment (027-1, 030-2).**

- **What is encoded.** After first-block postselection the window's state
  encodes `a_i = Y_1,i / ||Y_1||`, the increment's normalized amplitudes.
- **The leading factor.** For an absolute coordinate tolerance
  `epsilon_coord` and a known `||Y_1||`, the idealized leading factor is
  `31 pi ||Y_1,j|| / epsilon_coord` per batch.
- **The ratio in the receipt.** `||Y_1,j|| / gamma_j` runs from `0.235` to
  `0.692` over the 20 windows. The first window has `gamma = 2.2831` and
  `||Y_1|| = 0.5367`, as 027 states.
- **The recount.** From the stored per-window values, with
  `epsilon_coord = 2e-4` (an illustration: a standard deviation is not a
  tolerance):

  | quantity | published | corrected |
  |---|---|---|
  | readout calls | `2.37e7` (20 batches) | `1.20e7` (21 batches) |
  | leading-factor proxy, LCU / sparse, frozen forcing | `1.11e9` / `1.53e10` | `5.6e8` / `7.7e9` |
  | the same, uniform forcing numerator | -- | `6.8e8` / `9.5e9` |
  | ratio against route (b) | about 230 | about 110 to 140 (LCU) |

- **The ratio is not a comparison.** The two proxies omit different costs,
  so neither ratio is an established cost comparison.
- **On preparing `|x_r + Y_1>` (an amendment to 027).** The executed scheme
  never prepares it. The solver starts at `u = 0`, and `x_r + Y_1` is formed
  classically after readout. What is unpriced is each window's forcing state
  `|b_j>`, the coherent sign reference and the controlled solver.

**B. The executed protocol and the deterministic contract (027-2, 030-2).**

- **The protocol.** The executed protocol has 21 readout batches of 31
  coordinates at `R = 19`: 19 restarts (`r3_route_a.py:156`), plus
  independent output batches at `T = 4` and `T = 20` (`:161`). In general
  there are `R + 2` batches: 7, 12, 21 and 42 at `R = 5, 10, 19, 40`. That
  is the protocol priced; reusing the `T = 4` restart batch as an output
  would be a different protocol.
- **What the percentiles show.** Of the tested Gaussian levels, the four
  empirical marginal 95th percentiles over 50 draws are at most `1e-2` at
  `1e-4` and `2e-4`, and not at `3e-4` or above. That is a finite
  comparison of an unbiased, independent surrogate.
  - It is not a necessary or sufficient precision.
  - It is not a simultaneous 95 percent statement.
  - It carries about 10 percent sampling uncertainty at the boundary.
- **The shots branch** (`:77-86`) draws Gaussian noise with a Hadamard-test
  variance and the true norm. It is not binomial sampling or amplitude
  estimation.
- **The deterministic contract,** written out and not measured:
  - **Requirement.** Every one of `M = 651` coordinate estimates is within
    `epsilon_coord` simultaneously, with failure probability
    `delta = 0.05`. The 21 norm estimates give `M' = 672`.
  - **Tolerance split.** `epsilon_a = theta epsilon_coord / ||Y_1||` and
    relative norm tolerance `eta = (1 - theta) epsilon_coord / ||Y_1||`,
    with `0 < theta < 1` not selected.
  - **Union bound.** `ln(M'/delta) = 9.51`, or `ln(651/0.05) = 9.47`
    without the norm estimates.
  - **Calls per signed amplitude.**
    `c_AE (pi / epsilon_a) c_med ln(M'/delta)`. Here `c_AE = 2` (a signed
    amplitude from a Hadamard-test probability under the canonical
    estimation bound) and
    `c_med = 1/(2 (8/pi^2 - 1/2)^2) = 5.18` (the Hoeffding median boost).
    Together that is `98.6 / theta` times the idealized leading factor.
  - **Norm.** One estimate per batch, at
    `c_AE (pi / eta) c_med ln(M'/delta)` calls.
  - **Sign.** A controlled window solve, interfered with a reference basis
    state, per call. The controlled solver and the reference are unpriced.
  - **Sufficiency of `2e-4` is not established.** Worst-case errors add
    linearly over 21 batches. After the pulse `H` is a first integral of
    the closure, so batch `b` leaves an energy error of at most
    `epsilon_coord ||grad H(x_b)||_1` to first order. That sum is not in
    the receipts.
- **`R = 19`** is the only passing count of the four tested. It is not a
  unique optimum or a monotone law.

**C. The driven solver (027-3, 030-3).**

- **`beta_j`.** It was evaluated with the forcing frozen at `f_max`. The
  uniform numerator `Delta t (||c_0|| + f_max ||c_1||) / gamma_j` is 1.13 to
  1.62 times larger over the 20 windows (recount). It raises the LCU proxy
  from `5.6e8` to `6.8e8`. Neither expression is a solver guarantee until a
  theorem and an oracle are bound.
- **Normalization.** The sparse value `s x max|entry|` was measured on the
  `+f_max` generator only. A bound for the family `|f| <= f_max` needs both
  endpoints, since the entries are affine in `f`. The LCU value is a
  candidate triangle bound, with no circuit.
- **Growth constant.** It is that of the time-ordered propagator, not
  `sup ||exp(G_fmax t)||`.
- **Unresolved and listed.**
  - forcing-state preparation and forcing access;
  - solver discretization and precision;
  - propagator amplification;
  - readout recovery.
- **The `0.19 s` comparison** is one same-target benchmark, not an
  asymptotic lower bound or a no-speedup theorem.

**D. Route (b) (027-4, 030-4).**

- **What `4.9e6` is.** A leading-factor proxy: ideal readout factors times
  spectral-norm and time products, with the whole `1e-2` assigned to
  readout. A readout share `epsilon_read` scales the counts by
  `1e-2 / epsilon_read`.
- **What it omits.**
  - the Hamiltonian block-encoding normalization (`alpha_H >= ||H||`,
    unmeasured);
  - preparation of the correlated cutoff-16 ground state;
  - coherent simulation error;
  - norm and reference access;
  - total failure probability (four estimates, `ln(4/0.05) = 4.38`).
- **The cutoff differences.** Cutoff 16 against 20 gives `2.6e-4` and
  `9.6e-5`. These compare two finite models. They are not an
  infinite-cutoff bound or an implemented route's error, and they no longer
  fill route (b)'s error row.
- **The triangle inequality.**
  `|O_num - O_phys| >= |O_bare - O_phys| - |O_num - O_bare|`. It shows that
  accurate integration of the bare closure cannot remove its site-0
  discrepancy at the tested preparation and horizons. It does not show that
  direct simulation is the only physical route, or exclude corrected or
  hybrid formulations.

**E. The tube (027-5, 030-6).**

- **r2:252 had the inequality reversed.** It should read: "the largest
  `D_1/T`, with `D_1` the L1 norm defect of a comparison path and
  `e_0 = 0`, for which a radius `R` with `a + b R^2 <= R` exists
  (`a = Lambda D_1`, `b = Lambda T beta`)".
- **The remainder coefficient.** `beta = ||A_2||_2 = 1.995` is an external
  floating SVD estimate, not certified; it needs a validated upper enclosure.
  It would hold on all of R^31 if the field is exactly quadratic, for which
  the rounding-level cubic coefficients are evidence, not proof. These two
  premises are distinct from the uniform gain and the L1 defect (message
  `quantum-review-scope-20261018-004`, item 2 and addendum (a)).
- **What `13.22` and `54.10` are.** Conditional estimates, resting on four
  unbounded premises:
  - DOP853 taken as exact;
  - a sampled `max ||F||`;
  - a Richardson RK4 allowance;
  - `max |f'| = pi/2`, which r2 did not list. The rigorous bound
    `pi e^{-1/2} = 1.906` changes the inflation by about `1e-5`.
- **The grid values** `12.98` and `51.04` are lower bounds on the
  supremum.
- **What the helper shows.** The Codex helper reproduces the thresholds
  `4.48e-5` and `1.07e-7` and returns `diagnostic_only`.
- **No verified tube for route (a).** No verified L1 defect, restart-jump or
  readout budget exists, so no verified trajectory-error tube exists. A
  failed sufficient scalar criterion on supplied inputs shows only that; it
  is not the absence of an actual tube or a universal impossibility.
- **Wording.**
  - r2:261 "certified modulo those three" becomes "a conditional estimate".
  - r2:465 "RK4 step error `<= 2.3e-6`" becomes "RK4 step-halving
    difference `2.3e-6`, not an error bound".

**F. The common metric (027-6, 030-5).**

- **The conditional argument.** If the algebra generated by `A_1` and `D`
  contains an exact nonzero nilpotent `X`, no positive `G` makes both
  skew-adjoint: `G` would make `X` similar to a real skew-symmetric matrix,
  whose square has strictly negative trace unless it is zero.
- **What the numbers are.** The computed facts are numerical obstruction
  evidence, not a proof of exact nilpotency, exact membership or unbounded
  growth:
  - the floating Lie closure gives an algebra of dimension 91;
  - it contains an element with `||X^2||_F = 2.4e-13` (`4.4e-13`
    reduced);
  - `||exp(1000 X)|| = 647` (`679` reduced).
- **Applied where.** In r2 correction 19, r2:289, 300, 488, and in
  consolidation item 17.
- **Numerical ranks and spectral counts.** They carry the recorded scripts'
  tolerances.

**G. KvN (027-7, 030-6).**

- **The `T ||L||` row.** `(pi/h) sum_k max |F_k|` over searched box
  vertices (`2.0e7`, `3.9e8`, `7.3e10`) is a coefficient-wise access proxy,
  not a proved operator-norm lower bound. No realization or domain is
  specified.
- **Mesh and register.** The rules are heuristics, neither sufficient costs
  nor universal lower bounds.
- **Survivor conditioning.** At `sigma_0 = 0.1`, `T = 20`, 916 of 1,000
  antithetic pairs remain finite and under `1e4`
  (`c7_kvn_bias.py:85-87`). That row's bias, `20.7`, is a
  survivor-conditioned mean.

**H. Timing (027-T).** Each process's own timer was under 20 minutes. The
whole increment (preparation, reference, scoring, report, review, closure)
was not timed, so no whole-increment bound is established. This applies to
r2:122 and to pricing section 7.

**I. The scope of the conclusion (030-1).**

- **The headline becomes:** "On the L = 2 Holstein dimer, none of the routes
  assessed here shows a quantum advantage for integrating the archive
  equations under either contract."
- **The routes assessed:**
  - origin-centred and re-centred Carleman lifts, orders 1 to 4;
  - KvN;
  - Schrödingerization of the trajectory-linearized flow.

  All from the exact correlated preparation, for site-0 occupation and the
  Eq. (22) energy at `T = 4, 20`.
- **Not assessed.** NISQ costs, other reductions or encodings, and complete
  implementations.
- **"Any lift must be re-centred"** becomes "of the lifts tested, only the
  re-centred one reaches the contract".
- **r2:59-63 "Only one route attains it"** becomes "of the routes tested,
  only the re-centred lift attains it".

**J. The physical error (030-4).**

- **The bound.** For a fixed observable and pinned references, with closure
  discrepancy `Delta` and integration error `epsilon`, the physical error
  lies between `max(0, Delta - epsilon)` and `Delta + epsilon`. Cancellation
  is possible.
- **On site-0 at `T = 20`** (`Delta = 0.412`), a route fails the contract
  unless `epsilon > 0.40`.
- **Direct simulation** carries no closure discrepancy, but has no certified
  total allocation.
- **Applied to:** consolidation section 4 (the contract and the error
  against T-physical), section 5, withdrawn item 25, and pricing section 5.

**K. `J_min` and trace zero (030-5).**

- **`J_min`.**
  - `J_min = (F grad H^T - grad H F^T) / |grad H|^2` needs `grad H != 0`.
    Here `dH/dx_1 = -4 t_hop` is constant, which I checked from
    `closed_internal_energy`: `h_1 = -4`, and the quadratic part has no
    `x_1` row or column. So the gradient never vanishes, and the
    critical-point limitation does not bite.
  - It needs `F . grad H = 0`, which holds identically for the drive-free
    field only. Under the drive, `F . grad H` is the drive power.
  - It does not give Jacobi; the text already says it fails at all 101
    sampled states.
- **"No strictly stable complement (trace zero)."** This rests on the
  numerical spectrum, `|Re lambda| <= 2.2e-16` full and `5.4e-16` reduced,
  not on the trace alone.
- **Agreement with 030.** 030 notes that the critical-point extension is not
  a defect here. I verified `dH/dx_1 = -4` from the code.

**L. The packet-width thresholds (030-6, contested detail).**

- **What stays.** The thresholds `sigma_0 <= 0.020` (`T = 4`) and
  `<= 0.0024` (`T = 20`) remain protocol-specific one-point quadratic-law
  extrapolations, neither necessary nor sufficient.
- **The contested detail.** They are not survivor-conditioned:
  `c7_kvn_bias.py:108-115` takes them from the `sigma_0 = 0.1`, `T = 4` row
  and the `sigma_0 = 0.01`, `T = 20` row. In
  `r2/c7_kvn_bias.json`, `pairs_used` is 1,000 of 1,000 in both rows.
  Survivor conditioning affects only the `sigma_0 = 0.1`, `T = 20` row.
- **Root's resolution, recorded as a supported root-scope amendment.**
  - Root read the threshold source narrowly.
  - The rows `sigma_0 = 0.1`, `T = 4` and `sigma_0 = 0.01`, `T = 20` each
    use 1,000 of 1,000 pairs.
  - The `sigma_0 = 0.1`, `T = 20` bias remains survivor-conditioned, with
    916 of 1,000 pairs.
  - The code selects the largest `sigma_0` whose bias exceeds 3 standard
    errors. That is not an all-survivors rule.
  - The thresholds keep their limits: extrapolation, finite samples, and
    no guarantee.

**M. Counts, scaling and memory (030-7).**

- **Kept unchanged.** `2L^3 + 3L^2 + 3L - 1` (89 and 187) and "the fitted
  cost law predicts about `137 s`".
- **Polynomial cost.** A polynomial coordinate count makes one right-hand-
  side evaluation polynomial in `L`. It does not prove a polynomial
  integration cost in accuracy, horizon and conditioning.
- **The qubit count.** `2 log2 L + 4.1 L` qubits assumes:
  - one up and one down electron, densely encoded;
  - no spin-symmetry compression;
  - one local Holstein mode per site at cutoff 16 (`log2 17 = 4.09`).
- **Full-coordinate readout.** It costs the number of coordinates times
  `pi ||Y_1|| / epsilon_coord` per batch, which is not automatically the
  classical integration cost times an estimator overhead.
- **The 87 rows.** They come from `scaling-locality-20260920.md:259-261`:
  the memory law `74.8 (n (2 n + 1))^2` bytes of that implementation,
  applied to the historical 18 GB planning allowance.
  `agent_guidance/shared/memory-budget.md` (SHA-256
  `9a9cd10dc0f290f3c0111788f977a0c11d6443fc24de59ebffdaa1da1b0e43f3`)
  treats 18 GB as historical context and sets no fixed cap. So 87 rows is
  neither an intrinsic limit nor a campaign prohibition.

---

## Disposition of quantum-review-scope-20261018-004 (addendum)

| item | disposition | lines touched (this file) |
|---|---|---|
| (a) `beta` and the exact quadratic structure are external numerical estimates and rounding-level evidence, not certified; a failed sufficient scalar criterion means "no verified tube", not "no actual tube" | accepted | text E (the remainder coefficient and the route (a) tube). The same wording is carried into the consolidation's section 3.1 tube lines |
| (b) the KvN-bias threshold rows use 1,000 of 1,000 pairs; `sigma_0 = 0.1`, `T = 20` is survivor-conditioned (916); the selection rule is the largest `sigma_0` with bias above 3 standard errors | accepted, recorded as a supported root-scope amendment | the count paragraph; text L |

## Scope

- **Separately contracted work stays separate:** physical certificates,
  output and route selection, readout costs, and evidence adoption. No
  route, target or default is selected.
- **Ownership stands.** My module, test and measurement ownership is
  unchanged, and the 35-test result is preserved without a rerun.

## Receipts

All files are in `output/local_runs/paper_v_quantum_target_form_20261010/r4_flow_budget_review/`
(gitignored). They are arithmetic on stored receipts.

| file | SHA-256 |
|---|---|
| `r4_ledger_recount.py` | `b6b5b1373db2e7650cb7abd6170543ee8b0fcc7241187a46f923de471f591511` |
| `r4_ledger_recount.json` (source `r3_ledger_sigma2e-4.json`, `2cbb0bf9...3096`) | `5ffcc7e66c33b7bf01f791da34f65005fced9b9a7ad9677bf485a568beab3fe4` |
| `r4_helper_check.py` | `6fed1ab47a893f03b995a422535a38ce6a7757762e7ad162b71db3f80c8bf244` |
| `r4_helper_check.json` | `a038086ff2ee7973e79757a1e7e2eea2ca9d9b9b82b729c60cf5fb3a989d637c` |

Next action: the coordinator commits the two handoff files, the amended
pricing report and the amended consolidation.
