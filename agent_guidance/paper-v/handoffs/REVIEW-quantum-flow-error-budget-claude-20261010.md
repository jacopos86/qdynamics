# Independent review: the conditional flow-error budget helper (Codex increment 018/020)

Date: 2026-10-18. Author: the persistent Paper V quantum-integration agent
(Claude), as the existing B-Q3 author. Requested by message
`codex-to-claude-20261010-013` (paragraphs 4 to 6).

**Revision (2026-10-18, after `quantum-review-scope-20261018-004`).** The four
wording corrections of that message are applied in place to the 305-line
version root read (SHA-256 `c14ece20...5434`). The block at the end
dispositions each one with the lines touched.

**Status.** Exploratory and not promoted. Read-only on the Codex worktree. The
manuscript is not edited, no state-changing git command was run, and no
route, target, output, default or implementation is selected. Physical
trajectory certificates, output and route selection, and readout costs remain
separately contracted work.

---

## 0. Verdict

- **No defect in the derivation, in the scalar acceptance or in the evidence
  statuses.** The helper checks `a + b R^2 <= R` exactly, in rational
  arithmetic, for the binary floats it stores. It separates conditional
  assertions, sampled diagnostics and unsupported cases. It verifies no
  amplification, defect, remainder or domain bound of its own. It claims no
  physical trajectory certificate.
- **One conversion and documentation limitation, not a defect.** Under the
  accepted contract the helper checks normalized binary-float inputs. A
  non-float numeric input (for example a `Fraction` or a large `int`) is
  rounded to the nearest float, not outward. So a
  `conditional_supersolution` result can return a radius below the caller's
  exact non-float bound (`flow_error_budget.py:49`; case E in section 5).
  Exact non-float preservation was not promised, so this shows no defect
  for the represented binary inputs. No API, source or test fix is selected.
- **Fed my receipts, the helper reproduces my tube numbers and contradicts
  none of them.** It reproduces r2's admissible time-averaged defect,
  `4.4800e-5` (`T = 4`) and `1.0702e-7` (`T = 20`), and the radius bound
  `2 Lambda T delta`. It returns `diagnostic_only` for them, which is what
  they are.
- **One correction to the request's wording.** The values `12.98` and `51.04`
  are not certified pair suprema. They are grid maxima of `||Phi(t,s)||_2`,
  so they are lower bounds on the true supremum. The inflated values `13.22`
  and `54.10` are conditional estimates that rest on unbounded premises
  (section 2.3).

---

## 1. Inputs and hash verification

Every SHA-256 below was recomputed on 2026-10-18 before the file was read. For
the Codex files it was computed both on disk and on `git show 81710146:<path>`.
All of them match.

| input | revision | SHA-256 (recomputed) | lines |
|---|---|---|---|
| message `codex-to-claude-20261010-013.md` | -- | `b078b9a4df55ceb73e2c5ed3e8f7b695b89fd774ba7b2856cda045bedbcaea73` | 55 |
| `agent_guidance/paper-v/handoffs/quantum-error-bound-increment-20261010.md` (Codex) | `81710146` | `0d623d395c0074a6b86b0d988852c655dc0350cc92efb438954ff1c7fb8c6539` | 257 |
| `paper_5/src/paper5/stability/flow_error_budget.py` (Codex) | `81710146` | `c1e8aeda13f0453c14fb2afc89d83ef79eb73c6e15ce6b685fb0a63f38b42006` | 271 |
| `paper_5/tests/test_flow_error_budget.py` (Codex) | `81710146` | `a85827abddb691269c573e486b53b3b414128d001d7612fbac631c9d5c3b3851` | 351 |
| `agent_guidance/paper-v/handoffs/quantum-ode-theory-dossier-20261010.md` (Codex) | `81710146` | `40f6681a96920bcaa8605d4f29062370c2baedc12adc79b03a6e054e82590440` | 1,306 |
| root review `bq3-independent-review-20261010-020.md` | -- | `35e4de5316558a50c3ad12dd3e4de2a9e28748564845f2f7e4d2d975fcc88a66` (no hash was supplied; this matches the Codex report's received lock, line 19) | 26 |
| root review `quantum-r2-pricing-review-20261011-027.md` | -- | `3e6ecfc005219c5b21492a847680816f867c2ed56681b20ddc10ec9ba3784903` | 88 |
| root review `consolidated-results-review-20261011-030.md` | -- | `14c14fd4da0355840c944263d15cac0a6fb90fa6f683dad9fc533338f9442f3b` | 53 |
| my consolidation `quantum_integration_negative_result_20261011.md` | `424adca9` | `8534130ae1a04489b0979fb63f56ffc66d24d80d595168e3176fe1fa3e35f501` (working tree equal at reading) | 821 |
| my r2 `claude-architecture-quantum-integration-20261010-r2.md` | `475c47a7` | `e8c883567797d5fd7313ae9a386524d8c65ee697e0abdfca64fd7460702583bc` | 615 |
| my pricing `claude-quantum-route-a-pricing-20261011.md` | `d3c80e8a` | `e98c046f426d4a668a907e9f7002bad04e9c6ab07ddefd27262ab96a675ddee3` | 307 |

I read the dossier's delta from `f96678da` to `81710146` (122 changed lines) in
full. The unchanged parts are the `f96678da` text I had already read.

---

## 2. The derivation

### 2.1 What is verified

The setting: `x' = F(t, x)`, a base path `v` with defect `delta = v' - F(t, v)`,
`e = x - v`, `A(t) = D_x F(t, v(t))`, and `Phi(t, s)` its propagator.

**The scalar inequality.** Variation of constants gives
`e(t) = Phi(t,0) e(0) + int_0^t Phi(t,s) [N(s, e(s)) - delta(s)] ds`, where
`N = F(t, v+e) - F(t, v) - A e`. Suppose:

- `||Phi(t,s)|| <= G` for all `0 <= s <= t <= T`;
- `||N|| <= beta ||e||^2` on the closed tube.

Then `E(t) = ||e(t)|| <= G (e_0 + D_1) + G beta int_0^t E^2`. That is the
report's line 63. With `a = G (e_0 + D_1)` and `b = G T beta`, a radius
`R <= rho` with `a + b R^2 <= R` bounds `E` uniformly.

- **The first-exit argument (report lines 73-76) is correct.** For `a, b > 0`,
  `R >= a + b R^2 > a >= e_0`, so the start is strictly inside. At a first
  exit, `int_0^{t*} E^2 < T R^2` strictly, so `R < a + b R^2 <= R`, which is
  a contradiction. This also covers the double root.
- **The `b = 0`, `a = 0` and `T = 0` cases are correct.** With `b = 0` and
  `R = a`, the solution of the linearized equation stays in the closed tube,
  so by uniqueness it is the solution.
- **`R = 2a` suffices when `4ab <= 1`**, since `a + 4 a^2 b <= 2a`.
- **Including the identity at `s = t` forces `G >= 1`.** A terminal-time or
  fixed-start gain does not meet the contract. The piecewise test in
  section 4 is the right counterexample.

**`beta` is the remainder coefficient, not the Jacobian Lipschitz constant.**
The Riccati test `x' = x^2` checks this: `beta = 1`, where the Hessian value
would be 2.

### 2.2 Remarks: correct, and not defects

- **The acceptance condition is conservative.** Bihari's inequality applied to
  the same integral bound gives `E <= a / (1 - ab)` whenever `ab < 1`. The
  helper accepts only `4ab <= 1`, with radius `r_- = 2a/(1 + sqrt(1-4ab))`,
  and `r_- >= a/(1 - ab)`. Everything it accepts is valid. It rejects some
  cases with `1/4 < ab < 1` that a Bihari form would accept.
- **The path used for linearization can be either one.** With `e = v - x`,
  `e' = DF(x) e + N + delta`. So the same scalar inequality holds with `Phi`
  taken along the true solution `x`, and the defect belonging to the
  comparison path `v`.
  - r2 section 2.3 is that variant: `Lambda` along T-bare, taken as the
    exact closure solution.
  - The helper's docstring names only the first form (`A` along the path
    with the defect). A caller must say which path carries `A` and which
    carries `delta`.
- **If the closure is exactly quadratic, no finite tube domain is needed.**
  In that case `N(t, e) = A_2(e, e)` for every `e`, so
  `||N|| <= ||A_2||_2 ||e||^2`. The field is then a polynomial, smooth with
  local uniqueness on any tube.
  - **Structure.** The exact quadratic identity is itself a premise. The
    cubic coefficients are zero only to rounding (`1.5e-16`, task 1), and
    rounding-level values do not alone prove it.
  - **`beta`.** `beta = ||A_2||_2 = 1.995` is a floating SVD value
    (`r2/c3_pair_lambda.json`, `A2_norm`). It is an externally supplied
    numerical estimate and needs a validated upper enclosure before any
    certification.
  - **Four distinct premises.** The structure and `beta` premises are
    separate from the uniform-gain and L1-norm-defect premises; all four
    remain open.

### 2.3 The premises behind r2's `Lambda` (supplements r2:255-261)

r2 lists three unbounded premises. Two items need adding:

- **The DOP853 base trajectory is taken as exact.** It has no
  continuous-interpolant defect bound.
- **`max ||F||` is a sampled maximum:** `1.537` (`T = 4`) and `2.199`
  (`T = 20`).
- **The RK4 allowance is a Richardson estimate:** `3.2e-11` and `1.9e-8`.
- **`max |f'| = pi/2` is a fourth premise, not listed in r2.** A rigorous
  replacement follows from `|f'(t)| <= (pi/2)(1 + t^2) e^{-t^2/2} <= pi e^{-1/2} = 1.906`
  (using `|sin u| <= |u|`). It changes `mu_bar` by `0.0025 x 0.335 = 8.4e-4`,
  which changes the inflation factor by about `1e-5` relative.
- **Floating-point arithmetic is not enclosed.**

---

## 3. The code: what it checks, and what it does not

| # | finding | class | path:line |
|---|---|---|---|
| 1 | `a`, `b` and `Delta = 1 - 4ab` are formed in `Fraction` arithmetic from `Fraction.from_float` of the stored floats, so they are exact | verified | `flow_error_budget.py:240-246` |
| 2 | A negative exact discriminant is rejected with no tolerance or clamping | verified | `:156-158`, `:256-257`; test `:189-195` |
| 3 | The stable small root `a / ((1 + sqrt Delta)/2)` is computed at Decimal precision 80; then up to eight upward adjacent floats are each checked exactly against `a + b R^2 <= R` and `R <= rho` | verified | `:159-173` |
| 4 | Fallback: the stated tube radius itself is accepted only if it satisfies the inequality exactly | verified by reading; no test reaches this branch | `:174-178` |
| 5 | Status. `not_supported` for: a nonuniform domain, insufficient horizon, `G < 1`, a conditional input without assumptions, derived overflow, a negative discriminant, or no radius inside the tube. Otherwise `diagnostic_only` if any input is sampled, else `conditional_supersolution` | verified | `:230-265` |
| 6 | The checks on inputs are metadata only: enum types, horizon coverage, nonempty text, finite and nonnegative values. No input is compared with a trajectory, a field or a sample. So it verifies no external amplification, defect, remainder or domain bound | verified | `:45-59`, `:75-84`, `:95-100`, `:117-126`, `:208-239` |
| 7 | No physical trajectory certificate is claimed: "Sampled or terminal gains do not establish this theorem. The helper never verifies those inputs"; `CONDITIONAL` means the scalar check only | verified | `:8-16`, `:200-205`; report `:3-6`, `:131-132` |
| 8 | Inputs are normalized by `float(value)`. A `Fraction`, a `Decimal` or an `int` above `2^53` is rounded to nearest, so an upper bound can be stored below the caller's non-float bound (and a tube radius above it). Case E: `e_0 = 1/3` returns `conditional_supersolution` with radius `0.3333333333333333 < 1/3`. The accepted contract checks normalized binary-float inputs and did not promise exact non-float preservation. No API, source or test fix is selected | conversion and documentation limitation; not a defect for the represented binary inputs; no test covers it | `:49` |
| 9 | A `not_supported` result can carry a radius and `scalar_check = "verified_for_binary_inputs"`. Case B: a terminal gain returns radius `9.34e-5` with that label. The report says callers must inspect the status (`:120-121`), but the label reads like a pass | plausible usability hazard; documented | `:269-271` |
| 10 | Restart or repair jumps are excluded, as stated. An impulse `J_k` at time `s_k` would enter the same bound as `D_1 + sum_k ||J_k||`, since `||Phi(t, s_k) J_k|| <= G ||J_k||`. That would bear on route (a)'s readout restarts. It is not implemented, and it is not selected | plausible-unverified extension | `:198` |

---

## 4. What the 88 tests would and would not catch

**Operational deviation: I re-ran them.** I copied the module and the test
file, bytes identical (the hashes above), into a stub `paper5.stability`
package in my session scratchpad. With `PYTHONDONTWRITEBYTECODE=1` and
`-p no:cacheprovider`, `import paper5.stability.flow_error_budget` resolves
to the copy and gives 88 passed in 0.04 s. Nothing was written to the Codex
worktree.

- **The deviation.** The repeat was outside the no-repeat instruction of
  message 013 and its acknowledgement. It cannot establish fresh execution
  acceptance, and it waives no hold.
- **What is preserved.** The historical accepted result, 88 passed in
  1.50 s at `81710146`.
- **What follows.** No further test or measurement will be run.
- **Timing.** The 0.04 s is a process timer, not whole-increment timing.

The count of the parametrized cases, read from the file, is
`2+1+2+1+1+1+1+1+1+1+5+5+2+1+1+1+1+1+28+4+1+14+4+3+5 = 88`.

**They would catch:**

- a tolerance accepting an over-critical discriminant (`:189-195`);
- catastrophic cancellation in the small root (`:147-157`);
- accepting a rounded sum (`:160-167`);
- a factor-of-2 error between `beta` and the Hessian (`:130-144`);
- a missing `G` on the defect, which would fail `a = G (e_0 + D_1)` in
  `:114-127`;
- promotion of a sampled input (`:204-210`), and missing assumptions
  (`:213-217`);
- terminal or fixed-start gains (`:220-224`, `:252-264`), short horizon
  coverage (`:227-229`) and `G < 1` (`:232-235`);
- Boolean, nonfinite or negative inputs, wrong types and derived overflow;
- radii that miss the exact analytic error in the linear, nilpotent,
  piecewise and Riccati examples.

**They would not catch:**

- the rounding direction of non-float inputs (row 8 above);
- the fallback branch `:176-178`;
- a sampled input combined with an infeasible discriminant. The code
  returns `not_supported`, which I checked as case F;
- an amplification bound in a different norm from the defect or the
  remainder. That cannot be checked, since the norm is a caller's text;
- any physical system, time-dependent quadratic field or archive trajectory.
  All the examples are scalar or 2 by 2, with constant or piecewise-constant
  coefficients;
- the correctness of any caller's external bound, by design.

---

## 5. The helper fed my receipts

**How it was run.**

- **Inputs and outputs.** The driver is
  `output/local_runs/paper_v_quantum_target_form_20261010/r4_flow_budget_review/r4_helper_check.py`.
  It imports the copy `flow_error_budget_copy_c1e8aeda.py`, with the same
  SHA-256 as the Codex module, by path. Its inputs are the stored
  `r2/c3_pair_lambda.json`, G2 `s1.json` and G2 `s4.json`; their hashes are
  in the receipt. The receipt is `r4_helper_check.json`.
- **The settings fed to the helper.**
  - `beta = ||A_2||_2 = 1.995435`, entered as a conditional input carrying
    the structure assumption of section 2.2. It is a floating SVD estimate,
    not a validated enclosure.
  - `e_0 = 0`.
  - The norm is Euclidean on R^31.
  - The gains are marked sampled.
- **Cost.** It is arithmetic on stored receipts, with no new trajectory,
  propagator or noise run. It took `0.9 ms`.

| case | horizon | gain fed (its source) | `D_1` fed | helper status | radius or reason | my number | agrees |
|---|---|---|---|---|---|---|---|
| A | 4 | `13.2206` inflated `Lambda` (c3) | `T x 0.999999 x 4.4800e-5` | `diagnostic_only` | `4.7335e-3` | r2 `delta_avg = 4.48e-5`, radius `<= 2 Lambda T delta = 4.738e-3` | yes |
| A | 4 | same | `T x 1.000001 x 4.4800e-5` | `not_supported` | negative exact discriminant `-1.0e-6` | the threshold | yes |
| A | 20 | `54.0998` inflated (c3) | `T x 0.999999 x 1.0702e-7` | `diagnostic_only` | `2.3135e-4` | r2 `1.07e-7`, radius `<= 2.316e-4` | yes |
| A | 20 | same | `T x 1.000001 x 1.0702e-7` | `not_supported` | `-1.0e-6` | the threshold | yes |
| A | 4 / 20 | grid maxima `12.9763` / `51.0355` (c3) | at 0.999999 of the threshold | `diagnostic_only` | thresholds `4.650e-5` / `1.2025e-7` | r2 table column "terminal" at `T = 4` (`4.65e-5`) | yes |
| B | 20 | terminal `sup_s ||Phi(20,s)|| = 39.30`, domain `terminal_only` | `T x 0.5 x 2.03e-7` | `not_supported` | "amplification must cover every 0 <= s <= t <= T" | r2: the terminal quantity is wrong and 1.9 times too permissive (`2.03e-7` against `1.07e-7`) | yes |
| C | 4 | `exp(int mu+) = e^{7.108} = 1222` (G2 s1, T-bare) | at the threshold | `diagnostic_only` below, `not_supported` above | threshold `5.24e-9`, `1.2e-4` of case A | r2 section 2.8 exponent `7.11` | yes |
| C | 20 | `exp(int mu+) = e^{37.26} = 1.5e16` | at the threshold | same | threshold `1.4e-36` | r2 `37.26` | yes |
| C | 4 / 20 | `exp(T sup mu) = e^{7.383}` / `e^{46.13}` | at the threshold | same | `3.0e-9` / `2.7e-44` | r2 `7.38` / `46.13` | yes |
| D | 4 | exact-trajectory pair supremum `12.110` (G2 s1, sampled) | specified `D_1 = ||xi(4)||/G = 0.7361/12.110 = 0.0608` (G2 s4) | `not_supported` | `4ab = 284.6` at these inputs | the closure is off the exact trajectory by `0.028` (site0) at `T = 4` | consistent |
| D | 20 | `19.247` (sampled) | specified `D_1 = 1.8016/19.247 = 0.0936` | `not_supported` | `4ab = 5,535` at these inputs | `0.41` (site0) at `T = 20` | consistent |
| E | 1 | `G = 1`, `e_0 = Fraction(1,3)` | 0 | `conditional_supersolution` | radius `0.3333333333333333 < 1/3` | -- | shows row 8 |
| F | 20 | `54.10` sampled | `20 x 1e-3` | `not_supported` | negative discriminant takes precedence over the sampled reason | -- | correct |

**Reading.**

- **The r2 tube numbers are reproduced** (case A). Exactly at the threshold
  the outcome depends on the last float bit. It returned `diagnostic_only`
  at `T = 4` inflated and `T = 20` grid, and `not_supported` at the other
  two, with exact discriminants of order `1e-16`. That is the documented
  double-root behaviour.
- **The log-norm route gives absurd tubes** (case C). The Euclidean
  log-norm envelope is a valid uniform two-time gain if `mu` is an interval
  upper envelope. It admits a time-averaged defect 4 to 37 orders of
  magnitude below the pair supremum's. This agrees with r2 section 2.8:
  the recovery factor is `1.6e3` and `1.1e20`.
- **The sufficient scalar criterion fails for the specified diagnostic
  inputs** (case D).
  - **The inputs.** The gain is the sampled exact-trajectory pair
    supremum, with `beta = 1.995`. `D_1` is set to `||xi(T)|| / G`, from
    the G2 forced linearized response `xi(T) = int Phi(T,s) s(s) ds` to the
    closure defect.
  - **The result.** The discriminant is negative: `4ab = 285` and `5,535`.
  - **What it is not.** This is failure of a sufficient criterion on those
    inputs. It is not the absence of an actual trajectory tube, a statement
    about every valid input, or a proof of physical error.
  - **What a broader exclusion would need.** Validated lower bounds on the
    gain, on the remainder coefficient and on `||xi(T)||`. None is supplied.
  - **Consistency.** It does not contradict the observed closure
    discrepancy.

---

## 6. Review 020 reconciled with r2

Line numbers are r2 at `475c47a7`. "Disposition row" refers to
`DISPOSITION-quantum-027-030-claude-20261010.md`, which carries the
replacement text. None of 020's corrections is contested.

| 020 item | correction | status against r2 | number |
|---|---|---|---|
| 1 continuous bounds | upper log-norm envelope over each interval, covering drive and trajectory; interpolation validity; remainder domain | **partly addressed** at r2:242-244 (`mu_bar = max mu_2 + 2 ||A_2|| dt max||F|| + ||D|| dt max|f'|` bounds the Jacobian variation between samples) and r2:255-261. **Still open:** sampled `max||F||`; DOP853 taken as exact, with no interpolant defect; the unlisted `max|f'| = pi/2` (section 2.3). If the field is exactly quadratic the remainder needs no finite tube domain; that structure and a validated enclosure of `beta = 1.995` are separate open premises (section 2.2) | inflation `1.0188` / `1.0600`; `max||F||` `1.537` / `2.199` |
| 2 uniform amplification | `sup` over `0 <= s <= t <= T`, endpoint gaps, all injection times; no extension of sampled `P_1 U(T,s) P_K` by a full-step factor | **addressed** for `Lambda` at r2:113 (correction 25) and r2:230-261. The restricted-block quantities stay sampled with no between-sample extension claimed (r2:91, 155-156). **Open** only as a certificate | `12.976` / `51.035` grid; `13.22` / `54.10` inflated; `3.34` / `2.10` sampled observable bounds |
| 3 consistency against certification | RK refinement, tail identity and adjoint are consistency checks | **addressed** at r2:89 (correction 1) and r2:261 ("not a certificate"). **Still open** in wording: r2:261 "certified modulo those three" and r2:465 "RK4 step error `<= 2.3e-6`" (a step-halving difference). Amended in disposition row 027-5 | Simpson `1.1e-9` to `4.1e-9`; step-halving `2.3e-6` |
| 4 tube | `a + b R^2 <= R`, `a = Lambda (e_0 + int ||delta||)`, `b = Lambda T beta`; honest root arithmetic; an L1 norm defect, not a signed average; a reference defect is not a quantum algorithm's defect | **the numbers are addressed**: the helper reproduces the thresholds and the radius (section 5, case A). **The wording is open**: r2:251-253 writes the inequality reversed ("`r <= a + b r^2` has a solution"), and "time-averaged defect" must mean `D_1/T`, with `D_1` the L1 norm defect of the comparison path. Amended in disposition row 027-5 | `4.48e-5` / `1.07e-7`; radius `<= 4.74e-3` / `2.32e-4` |
| 5 spectrum | a Ritz residual is not a global extremum; a separate `sigma_min` lower bound is needed | **addressed** at r2:462 and r2:583 (no certified `sigma_min`) and r2:405-407 (grid maxima are lower bounds). **Open:** "an upper bracket within 0.003 percent" (r2:462) is ARPACK-internal, not independent, and the spectral and rank statements in r2:283-291 are numerical with tolerance | `kappa_stack` `12,071`, `166,189`, `11,558,518` |
| 6 KvN | `A = -(F.grad + div F/2)` skew, `H = iA`; keep the half-divergence for frozen sub-blocks | **addressed** for every coordinate sub-block: each diagonal Jacobian entry vanishes identically (r2:116, correction 28), so any frozen coordinate block has zero divergence. **Open:** the 28-coordinate reduction is a linear change, not a coordinate block. Only its linear part's trace was measured (`3.8e-17`, r2:286). The removed directions (a rotation of `B_+`, and the conserved `ell`) make zero divergence plausible but unproved. The operator realization (domain, boundary flux) is unassessed | `0.0` diagonal entries; `3.8e-17` |
| 7 retained Gram | raw moments are degree two, so the retained Gram is linear in `[1, x, x (x) x]`; block 2 of a truncated lift can differ from the square, and its error needs a bound | **addressed** at r2:99 (correction 11), r2:98 (correction 10) and r2:469-471. **Open:** a propagated bound for the Gram and observable error from a truncated block 2; only the discrepancy is measured | Frobenius `11.3` to `802`; reconstruction `3.1e-16` |
| 8 outputs | `||x|| ||c||` linear and `||x||^2` quadratic readout; preparation, coherent phase or sign, norm, postselection, amplification, confidence; `||B||^2` in phonon number; spin convention; truncation error of lift-block energy | **partly addressed** at r2:338-376 (sign crossings, scale, `pi ||c|| ||x|| / epsilon`, union factors), r2:98 (`|B|^2 + Tr N`) and r2:131 (per spin). **Open:** norm-estimation and coherent-reference costs (the interference of r2:347-348 is coherent, but its preparation and the controlled solver are unpriced); the `||x||^2` factor for `H`; a confidence contract for route (a)'s 651 estimates; a bound, rather than a measurement, for the lift-block energy truncation. Explicit costs are in disposition row 027-2 | `x_0/2` changes sign 7 times to `t = 20`; `|B|^2` 79 to 88 percent; block-2 energy `0.178`, `0.029`, `0.0094` |
| 9 negative findings | numerical nullspaces are not certificates; restrict to the tested classes; finite threshold events are not all-time bounds | **partly addressed** at r2:102-104 (corrections 14-15 scope the Poisson and reduction negatives). **Was open, now corrected:** r2:107 (correction 19) called the nilpotent witness "a proof"; it is numerical (disposition row 027-6). The escape "at `t = 141.4059`" (r2:485-486) is a `1e4` threshold crossing, not a blow-up proof | `||X^2||_F = 2.4e-13` |
| 10 physical error | an all-coordinate RMS is not an observable floor; price functionals; restrict the horizon and norm-ratio statements | **addressed** at r2:50-63 and r2:330-336 (observable-specific model error). **Open:** r2:321-324 ("rescaling helps", "order 4 buys accuracy only for `t <= 2`") hold at the tested preparation and orders only | site0 `0.0284` / `0.412`; `H` `0.0150` / `0.0150` |

---

## 7. Scope

- **Separately contracted work stays separate.** That covers the physical
  certificates (a continuous interpolant defect, a rigorous uniform two-time
  gain, numerical enclosures), output and route selection, and readout
  costs. No route, target, output, default or reference is selected here.
- **No new measurement.** The only executions were:
  - the helper on stored receipts;
  - the Codex tests on bytes-identical copies, an operational deviation
    recorded in section 4;
  - the arithmetic recount for the disposition
    (`r4_ledger_recount.py`, stored per-window values).
- **Ownership is unchanged.** My module, tests and measurements stand, and
  the 35-test result is preserved without a rerun.
- **The Codex worktree was only read.** Its files were read, hashed and
  copied.

## 8. Receipts

All files are in `output/local_runs/paper_v_quantum_target_form_20261010/r4_flow_budget_review/`
(gitignored).

| file | SHA-256 |
|---|---|
| `flow_error_budget_copy_c1e8aeda.py` | `c1e8aeda13f0453c14fb2afc89d83ef79eb73c6e15ce6b685fb0a63f38b42006` |
| `r4_helper_check.py` | `6fed1ab47a893f03b995a422535a38ce6a7757762e7ad162b71db3f80c8bf244` |
| `r4_helper_check.json` | `a038086ff2ee7973e79757a1e7e2eea2ca9d9b9b82b729c60cf5fb3a989d637c` |

## Disposition of quantum-review-scope-20261018-004

Message SHA-256 `386c7b88964d0a6dd1b77820e6c71d4b48cbdc15717e6a1011519518098f663d`
(13 lines), verified before reading. Every item is accepted and none is
contested. All changes are wording; nothing was run for them. Line numbers
are in this revised file.

| item | correction | disposition | lines touched |
|---|---|---|---|
| 1 | Case D shows only failure of a sufficient scalar criterion on the specified diagnostic inputs; it is not the absence of a tube, a universal input impossibility or a physical-error proof | accepted | 255-256 (case D rows: specified inputs, `=` in place of `>=`), 272-285 (the reading) |
| 2 | Exact quadratic structure would remove a finite tube domain, but `beta = 1.995` is a floating estimate needing a validated enclosure, and rounding-level cubic coefficients do not alone prove the structure; keep these premises distinct from the gain and the L1 defect | accepted | 124-137 (section 2.2), 235-237 (section 5 settings), 297 (020 item 1). The same wording was applied to the disposition file's text E, and to the consolidation's section 3.1 tube line and section 1 heading |
| 3 | The `Fraction`/`Decimal` case is a conversion or documentation limitation under the normalized binary-float contract, not a defect for represented inputs; no fix selected | accepted | 28-35 (verdict), 167 (row 8) |
| 4 | Record the repeat of the 88 Codex tests on scratch copies as an operational deviation outside 013's no-repeat instruction; it gives no fresh acceptance and waives no hold; preserve the historical 88-test result; no further runs; 0.04 s is not whole-increment timing | accepted | 175-188 (section 4), 318-319 (section 7) |
| -- | the dated revision line | added | 7-10 |

Next action: the coordinator commits this file and the disposition with its
two in-place amendments.
