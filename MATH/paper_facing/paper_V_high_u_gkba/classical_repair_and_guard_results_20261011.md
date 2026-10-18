# The archive closure's divergence, the cheap repair, the guard and their hybrid: classical results to 2026-10-11

**What this file is.** A consolidated summary of the classical results of
2026-10-10 and 2026-10-11 on branch `claude/paper-v-routes-20260914`.

- **Status.** It is exploratory and not promoted. No number here is adopted
  evidence. Every receipt that records a classification says
  `exploratory_local_not_promoted`.
- **Decisions that stay with the user.** Adopting evidence, promoting it,
  demoting it, and inserting anything into the manuscript
  (`MATH/paper_details/paper_V_high_u_gkba.tex`, untouched by this work).
- **What changed.** No default changed, and no module under `paper_5/src`
  was edited.
- **Supersession.** For everything it covers, this file supersedes
  `MATH/paper_facing/paper_V_high_u_gkba/cheap_repair_route_results_20260918.md`.
  That file is left unedited as the record of 2026-09-14 to 2026-09-18.
- **What was run.** Every run here is the Holstein dimer at
  `lambda_ep = 1.5`, `gamma = 0.5`, with no Hubbard `U` (`DimerParameters`
  has no `U` field). That is the manuscript's Holstein-dimer smoke case,
  which it places before the high-`U` stress tests (tex line 344). No
  high-`U` case was run.

**Revision note 2 (2026-10-18, dispositions of root review 030 (classical
part) and review 031, with the persisting points of reviews 028 and 029).**

- **What changed.** Revision 1 is commit `2c096bad` (SHA-256 `332dbbc3...`).
  Amended in place, with no new run.
- **Where each change is recorded.** Each line below gives one changed or
  added statement and its line in this revision. The finding behind each
  is a row of
  `agent_guidance/paper-v/handoffs/DISPOSITION-closure-028-031-claude-20261010.md`.
- **What stands.** The U-free scope, the sampling and timing
  qualifications, the task-11 censored exception and the metric-dependent
  exact-correlated hybrid gains, which review 031 records as resolved, are
  preserved.

- L94: first revision note: its line numbers refer to `2c096bad`; the numbers-check artifact was not preserved
- L202: sources: `DI` pinned; the exact divergence-free identity's source and the amended report added
- L292: the convergence statement scoped to the tested refinements
- L300: volume preservation cited to the exact identity; the floating-point traces and Lyapunov sum kept separate; the attractor statement qualified
- L324: 'no attractor' tied to the exact identity
- L327: 'transient chaos' stated as finite-time stretching; no global attractor or unboundedness claimed
- L354: 'nothing in the closed equations confines' scoped to the energy
- L456: section 3 heading: the retained-Gram necessary condition, not representability
- L458: the Gram and minors defined as necessary conditions, not physical representability
- L470: 'lost within units, from every state' scoped to the tested condition and states
- L490: 'discriminates nothing' replaced by a finite, censored, in-sample association
- L496: within-ensemble ordering stated as weak or absent in a finite, in-sample test
- L534: the clip's targets separated from what the capped map achieves, with residuals and receipt gaps
- L575: the energy-hold target: computed post-RK4 energy, not an accumulated drive-work reference
- L586: compactness stated as a conditional set bound, separate from the capped map; 'tight' becomes 'sharper'
- L603: the exact zero-energy trace witness `x17 = -omega n / (2 g)`; trace control needed, not uniquely
- L614: heating table: the largest recorded sampled value (records every 400 steps), not a supremum
- L619: heating: rising, not shown unbounded
- L625: 'which families are necessary' replaced by a finite ablation of the capped map
- L648: 'the correlation bounds are necessary' replaced by a finite ablation result
- L658: non-converged counts are batch-step counts; per-row keep-only status added
- L671: keep-only counts as batch steps, with the per-row residuals
- L695: triggered guard: bare before the first trigger; zero events are finite, censored, capped-algorithm observations
- L722: first order stated as an observation of the tested always-on maps
- L730: accuracy comparisons stated metric by metric
- L781: pulse-work attribution qualified
- L792: the critic's quotation attributed and its comparison limited
- L799: the Gram-cone and boson maps: no event in 600 units, not 'confine'; not projections onto an enforced intersection
- L811: starts satisfying the tested necessary constraints; 16 against 64 members; no independence claim
- L822: the projection comparison qualified (unmatched ensembles, provenance); 'a property of the flow' scoped
- L915: the `T = 4` Hartree--Fock contradiction corrected: improvement is metric-dependent, no arm improves all three
- L925: re-selection counts: 222 main-arm plus 12 refinement, all infeasible
- L932: skipped-selection cost: source-level support only; solver and RNG effects and bitwise equality not verified
- L941: cheap-repair ratio scoped to the measured protocols (8 of 10 failed re-selections)
- L950: the hybrid's failed gates: negative post-event Gram, 99 of 234 events reduce the violation, the exact-correlated `Delta = 20` cap without a residual receipt
- L961: what the hybrid receipts prove: 13 arms to 60 with no resume, 117 nonfailure escape statuses; one refinement; whole-increment gate unestablished
- L996: reading: stiffness and energy drift scoped to the diagnostics
- L1003: reading: backward-Euler conditioning is not a general nonstiffness certificate
- L1006: reading: 'unbounded growth' replaced by finite growth past the threshold; finite-time stretching
- L1030: reading: positivity stated as the tested necessary condition
- L1035: reading: the joint loss 'does not predict' scoped to the tested, censored ensembles
- L1038: reading: energy drift and stiffness not excluded in general
- L1047: reading: what the capped clip achieves, separated from the conditional compact-set bound
- L1053: reading: compactness conditional; capped enforcement and the 600-unit horizon stated
- L1093: reading: 'volume-preserving' cited to the exact identity
- L1152: withdrawn-claim 14: 'tight' becomes 'sharper'
- L1223: withdrawn claims 35 to 43 added (revision 2)
- L1252: unmeasured list: per-process limits are not the 1200-second whole-increment gate; nothing is selected
- L1279: the exact correlated estimates labelled as extrapolations of the Hartree--Fock profiles
- L1275: item 2's hours stated as many bounded subincrements
- L1332: unmeasured items 15 and 16: execution provenance and the binding prerequisite holds

**Revision note (2026-10-11, after numbers check).** The first version is
commit `3a9981f7` (SHA-256 `714d34a8...`). A numbers check traced 147 of its
194 statements and found 47 mismatched; it also listed missing items. Every
fix was applied in place, with no new run. Line numbers below refer to this
revision. Where a source is inconsistent with itself, the line says which
value is used and why. I disagree with none of the fixes. The line numbers
in the list below refer to that revision (commit `2c096bad`), not to
revision 2. The numbers-check artifact itself (the coordinating session's
`classical_summary_verify.json`) was not preserved, so its counts are
author-reported and cannot be pinned.

- L17: added: every run is the `U`-free Holstein dimer smoke case; high `U` was not run (tex 344)
- L186: added: exact 95 percent upper bound for zero counts, `0.31` for 0 of 8 and `0.17` for 0 of 16
- L199: `0.22` is now the spread (`54.345` to `54.565`), not a largest change
- L201: added: Radau `434.472`
- L207: added: the tolerance and method shifts (`0.29`, `0.0067`, `0.07`, `0.28`) and what they measure
- L230: energy: twelve sampled times, after the pulse for the `t = 0` starts, not recorded past the event
- L254: added: the `t = 0` exponents (`0.3089`, `0.1558`); `0.047` is for the `t = 60` start only
- L274: Lyapunov-vector shares: horizon `t = 70` to `425.58`, phonon-covariance `0.147` added
- L314: `214.8` is the same recorded run's first five segments (`4,296.9` s, `DI` 4.1), not a different run
- L315: session walls now carry horizon, session, load and the minimum-of-three qualifier
- L318: repair time `19.0` s from the receipt (`R §13.11` prints the second run's `18.8`)
- L324: added: the `R §14.4` breakdown
- L326: scoped to the repaired chain's states
- L338: added: the unperturbed trend rests on three states of one chain
- L342: added: integrator sensitivity (`12.3`, `-2.2`)
- L347: added: task 11's `t = 20` state as the exception
- L362: added: growth rates `0.11` and `0.045`, amplification `47.9` and `26.4`, the nonlinear difference
- L377: 'already negative at the start, or turns negative within'
- L384: added: the repaired states' own Gram and the chain's 87 / 29 percent (every `0.25`) against 24.2 percent (every `0.0025`)
- L390: added: the depth-threshold leads (`175` to `318`, `14.5` to `51`, `14.5` to `724.2`)
- L401: within-ensemble range `-0.04` to `0.25`, with p and the five-ensemble scope
- L406: recorded: `R §13.3` prints `-0.01`; the receipts give `-0.0443`, used here
- L412: added: the early crossings' weak ordering (`-0.06` to `0.42`)
- L416: coverage: the leads are for section 10's five ensembles; 9 of 23 thresholds is the table's coverage
- L428: equal-skill lead `1.2` to `6.2` from the receipt (`R §13.3` prints `1.3`)
- L430: added: Gram `< -5` leads the coordinate passing 4 by `1.7` to `6.5`
- L435: heading: with and without the energy held
- L448: Gram fractions: unit records, split by variant (energy hold 74 percent, `-0.366`; 97 percent, `-0.625`)
- L459: boson minimum: `-0.302` is the clip alone; `-0.233` with the energy hold
- L470: added: the 200-alternation cap and how non-convergence is counted
- L492: 'must be held exactly' replaced by 'must stay bounded above'
- L496: heating table: supremum over 3000 units labelled as such; median largest coordinate at 3000 added
- L502: added: compactness is analytical; measured confinement stops at 600
- L527: added: the three non-converged drop-one cells, the `t = 60` cell, the keep-only cells
- L549: added: the triggered guard's single activation at `t = 37.26`, unconverged at the cap
- L557: cost table: horizon, minimum of three rounds and session loads
- L572: heading: 'buys nothing' replaced by 'does not approach the cheap repair'
- L577: accuracy tables: `T = 8` column, `h` to `h/2` column and every `R §13.6` variant restored
- L614: added: the site-0 gain against the bare closure at `T = 4`
- L618: added: the energy loss from exact correlated (`0.018` to `0.160`; `0.035` with the hold)
- L629: added: the Gram-cone and boson maps' heating, Gram and boson minima
- L674: hybrid wall: horizon and session load stated
- L676: Gram sampling stated (grid samples against section 4's unit records)
- L697: repaired-chain wall: `t = 0` to `0.4` one run, and the recorded wall's other session
- L714: added: the other arms are 40 to 41 times worse (derived)
- L719: `4.5` corrected to `4.6` (receipt values)
- L723: 'does not improve on the guard' replaced from the table; `Delta = 5`, 10 and 20 beat it
- L731: Eq. (22): `0.0610` at `Delta = 1` and `0.0412` at `Delta = 2`
- L733: recorded: `R §14.0` item 2 is inconsistent with its own table; the table is followed
- L738: Hartree--Fock gain scoped to `T = 20` and 60; `T = 4` does not improve
- L747: re-selection: per-arm means, with single-event ranges
- L753: derived cost: exact correlated `Delta = 20` (`0.594`) excepted, and why its wall is higher
- L770: reading scoped to the `U`-free dimer, offered for the high-`U` slot by analogy only
- L785: closure wall: each session named (`0.139` to `0.140`; `0.130` to `0.132`)
- L790: energy: twelve sampled times through the event, not recorded past it
- L800: chaos `0.0468` tied to the `t = 60` start; the Hartree--Fock start is direct
- L805: 'every state tested' replaced by the tested states, with task 11's exception
- L814: positivity: 'already negative or within' wording, `0.75` to `1.5`, `t = 0.015`
- L819: added: joint representability, not the density matrix; `rho`'s own drift not timed (`-216.4` bare)
- L827: added: the placeholder's fifth option, self-consistency instability, addressed
- L830: added: the tex 347 invariants (`Tr rho`, Hermiticity, step rejections)
- L839: cost of confinement `1.55` to `3.07` times (`R §13.11`); clip alone `1.18` and not compact
- L852: 'at most `0.017`' replaced by the four derived changes
- L856: repair cost: four walls with sessions against `0.130` to `0.216`
- L860: 'every repaired state blows up' scoped, with task 11's exception
- L863: 'stays 24 times worse' scoped (others 40 to 41); exact correlated restated from the table
- L871: added: the quantum encoding slot (tex 353)
- L899: claim 5: 'bare' restored; exact-reference states inside the joint cone
- L909: added claim 9: `113.43` against a fresh seed at `110.5`
- L915: claim 11: restriction applies to task 13; task 11's 41 of 64 restored
- L926: claim 12: withdrawn `1.29` removed; median `366` to `469`
- L931: added claim 13: survivors' range restricted to the clip maps
- L934: claim 14: `4 sqrt 2 g` is valid but loose, not withdrawn
- L940: claim 15: scope `R §8` to `§12`; the restoration fails conditionally
- L948: added claim 17: the extension sentence restricted
- L951: claim 18: values labelled (`0.46` the clip, `0.20` bare); `1.08` superseded
- L956: added claim 19: '500 times' replaced by `470` to `1340` and `1.0e3` to `2.5e3`
- L960: added claim 20: operation count
- L975: claim 26: 'from `t >= 10`' restored; `t = 5` state at `421.4`
- L982: claim 28: `-0.04` with the `R §13.3` discrepancy noted
- L988: claim 29: 'median' and 'both tuned' restored
- L992: claim 30: 'or not at all' restored
- L1001: added claim 34: `R §14.0` item 2
- L1008: unmeasured list: every figure labelled an estimate with its measured source
- L1014: item 1: measured walls `158.5` to `272.1` s, so 3 to 5 minutes per batch
- L1029: added item 4: converged map onto the Gram cone and the energy surface
- L1043: item 6: under half a minute per arm, not minutes; Hartree--Fock qualifier
- L1052: item 9: section-9 walls (`28` to `444` s) and the `0.31` bound
- L1064: item 11: bare cost with sessions
- L1074: item 13: about 25 minutes, not seconds
- L1080: item 14: about 3.5 minutes at `Delta = 1`

**Sources.** No new run was made for this file or its revision. Every
number is quoted from the sources below. Where a number comes only from a
receipt, or is derived, the text says so.

| short form | file | commit |
|---|---|---|
| `R §0`--`§7` | `agent_guidance/paper-v/overnight/closure-instability-20261010.md`, sections 0 to 7 (the linearized flow) | `40771435` |
| `R §8`--`§12` | the same report, sections 8 to 12 (escape statistics, predictor, confinement) | `c1c1f9cc` |
| `R §13` | the same report, section 13 (corrections after three refuters and a critic, and the ablations) | `36b686b3` |
| `R §14` | the same report, section 14 (the guard plus sparse repair hybrid) | `8390a83b` |
| `R §13`, `§14` (amended) | the same report after the 2026-10-18 amendments of sections 13 and 14 (dispositions of reviews 028 to 031), SHA-256 `30187293aeb89b3e355599560930ba26c5a79cd4bd263413d61cdbcb0fe0c710` | uncommitted at this writing |
| `DI` | `agent_guidance/paper-v/overnight/divergent-initialization-20260920.md` (the repaired chain's own record), SHA-256 `87508b2c4e7ebf7d82b98fc2076c01a14090d1b79d3d8fe565d4c6737cf94e7e` | `37e4067d` |
| `QF` | `MATH/paper_facing/paper_V_high_u_gkba/quantum_integration_feasibility_20261010.md` (the exact divergence-free identity), SHA-256 `72d3414c425948351a29af06ce2b74e2b313519eee6d42f2cdd5f6827961a1a1` | `475c47a7` |
| `W` | `paper_5/notes/electron_phonon_closure_worklog_20260803.md`, blocks dated 2026-10-10 and 2026-10-11 | as above |

Where `R §8`--`§12` and `R §13` disagree, `R §13` holds, and section 7 below
lists the replaced sentence. Receipts are under
`output/local_runs/paper_v_instability_20261010/`, with `task2_corrections/`
for `R §13` and `task3_hybrid/` for `R §14`. The SHA-256 of every receipt is
in `sha256_task2.json` there (520 files after `R §14`). The manifest file's
own SHA-256 is
`767f6eec4c5facf11dc9948d88c3a9ad13907db8787bbbe98304ab88918d795f`.

**Configuration.** Unless a row says otherwise:

- the model is the Holstein dimer at `lambda_ep = 1.5`, `gamma = 0.5`
  (`omega_ph = 0.5`, `g = 0.6124`), drive amplitude 1, phonon cutoff 16;
- the "bare closure" is the unrepaired archive closure (`closed_scalar_rhs`,
  Eqs. (14a) to (14e) in the 31 closed scalar coordinates). Its blocks are
  the electronic one-body density matrix `rho` (3 coordinates), the coherent
  phonon amplitude `B` (4), the phonon covariance `N` (4), the anomalous
  phonon covariance `A` (6) and the electron--phonon correlation `C[q]` (14,
  of which 2 are the shared trace);
- the "`1e4` event" is the first time the largest absolute closed coordinate
  reaches `1e4`.

**The two preparations, never mixed.**

- The **Hartree--Fock preparation** is the Hartree--Fock electronic state
  with zero electron--phonon correlations: the record's divergent product
  state.
- The **exact correlated preparation** is the exact ground state of the
  truncated Hamiltonian.

Each has its own exact driven trajectory, and each is scored only against
its own:

- "the exact trajectory from Hartree--Fock" is the reference stored beside
  the repaired chain;
- "the exact trajectory from the exact correlated preparation" is the
  task-11 and task-13 reference.

**The accuracy metric.** "Accuracy over `[0, T]`" means the RMS over every
`0.0025`-grid sample in `[0, T]` against the exact trajectory from the same
preparation (`_accuracy_metrics`, `apcm_carried_witness_analysis.py`). It
is cumulative over the window, not the error at `T`. It is written as
`site-0 occupation / Eq. (22) energy / all 31 coordinates`. Each accuracy
row is one unperturbed trajectory.

**Escape tests.** An escape test hands a state to the bare closure. It uses
admissible perturbations: `x + eps u`, with `u` from
`constrained_initial_direction` (Frobenius norm 1). Such a `u` is tangent at
first order to the Eq. (22) energy and to Re and Im of the shared trace. No
positivity is imposed. The integrator is DOP853 at `rtol 1e-10`,
`atol 1e-12`.

**Counts and intervals.** No interval is attached to a count of zero
events. With 0 of 8 fired, the exact 95 percent upper bound on the
per-member probability of an event within the span is `0.31`. For 0 of 16
it is `0.17` (`R §13.12`, `§14.7`).

---

## 1. The divergence

1. **Event times of the bare closure.**

   | start | `1e4` event | source |
   |---|---|---|
   | Hartree--Fock preparation, `t = 0` | `t = 54.4613`; 64 perturbations of size `1e-4` spread it over `0.22` (`54.345` to `54.565`) | `R §1`, `§3.3`, `§6` |
   | exact correlated preparation, `t = 0` | `t = 141.4057` | `R §0` item 1, `R §1` |
   | the exact trajectory from Hartree--Fock, its `t = 60` state | `t = 433.8964` at DOP853 `rtol 1e-10` (the record's value); `434.1811` at `1e-12`; `434.1878` at `1e-13`; RK4 at `0.00125` gives `434.1160`; Radau at `rtol 1e-10` gives `434.472` | `R §0` items 1 and 2, `R §3.1` |
   | the repaired chain's `t = 60` state | `t = 410.7827` | `R §1` |

   - The exact correlated value corrects the record. The record's "never,
     through `t = 140`" (`DI` section 2.2) was a window shorter than the
     event time.
   - The `t = 60` event moves by these amounts:

     | change | shift |
     |---|---:|
     | `rtol 1e-10` to `1e-12` | `0.29` |
     | `1e-12` to `1e-13` | `0.0067` |
     | RK4 at `0.00125` to DOP853 at `1e-13` | `0.07` |
     | Radau at `1e-10` to DOP853 at `1e-13` | `0.28` |

     These shifts are the size an admissible perturbation of `1e-12` to
     `1e-10` produces (`7.1e-3` at `1e-12`, `1.31` at `1e-10`). They are the
     trajectory's sensitivity acting on integration error, not an
     integrator instability (`R §3.1`).
2. **The threshold crossings are robust to the tested refinements.** Every
   event time converges under halving and quartering the step, under
   tightening the tolerance, and across one explicit and one implicit
   method, at all three starts (`R §0` item 2, `§3.1`). The remaining
   dependence at the `t = 60` start is the table above. This supports the
   robustness of the observed crossings to those refinements. It is not an
   exact trajectory or error certificate, and not an exhaustive exclusion
   of numerical mechanisms.
3. **Volume and energy.**
   - **The exact identity.** The archive field is exactly divergence-free.
     The trace of its Jacobian vanishes identically, from its polynomial
     coefficients (`tr A_1 = tr D = 0`, `sum_i A_2[i, i, :] = 0`; `QF`,
     "The structural reason no expansion point can be dissipative"). By
     Liouville's theorem the smooth flow then preserves phase-space volume
     on its domain of existence.
   - **Floating-point observations, separate from that identity.** The
     evaluated trace is at most `6.1e-13` at every sampled state, against a
     Jacobian 2-norm of up to `16.4` in the bounded phase and `2.8e4` at the
     event. The 31 finite-time Lyapunov exponents sum to `-8.9e-8` over 374
     units.
   - On each bare trajectory the Eq. (22) energy is the same to six
     decimals at twelve sampled times, through the `1e4` event. For the two
     `t = 0` starts this holds after the pulse:

     | trajectory | energy |
     |---|---:|
     | `exact_t60` | `-3.214773` |
     | `exact_correlated_t0`, after the pulse | `-3.556387` |
     | `hartree_fock_t0`, after the pulse | `-3.260693` |

     During the pulse the drive does work. Past the event the energy was not
     recorded (`R §0` item 3, `§6`).
   - By the exact identity the smooth flow cannot contract volume, so it
     has no attracting set of lower dimension. That follows from the
     identity, not from the floating-point traces.
4. **Positive finite-time stretching from the exact `t = 60` state
   (transient chaos on these diagnostics).** Positive finite-time exponents
   and finite coordinate growth are observations. They do not prove a
   global chaotic attractor or mathematical unboundedness. Along the bare
   closure from the exact Hartree--Fock trajectory's `t = 60` state, the
   largest finite-time Lyapunov exponent over `t = 60` to 400 is `0.0468`
   (Benettin, step `0.0025`; `0.0463` and `0.0467` at steps `0.01` and
   `0.005`; `R §3.2`). The tangent-linear QR spectrum gives `0.047` at
   `T = 140`.
   - 64 perturbations of size `1e-4` fire after `293.1` to `478.1` units,
     median `351.7`, with none before `293` (`R §3.3`).
   - The bounded phase rides on a slow growth of the orbit's amplitude:
     the ten-unit maximum of the largest coordinate is `1.56`, `2.36` and
     `3.14` at `t = 60`, 250 and 400.
   - From the two `t = 0` starts the growth is direct, with no long bounded
     phase. The mean Benettin exponent is `0.3089` over `t = 0` to 54 from
     Hartree--Fock and `0.1558` over 0 to 141 from exact correlated (step
     `0.0025`; `R §3.2`).
5. **The escape goes through coordinates the Eq. (22) energy does not
   contain.**
   - The energy is linear in the phonon covariance and in the diagonal of
     the electron--phonon correlation (`omega_ph Tr N` and
     `4 g Re sum_q C[q]_{qq}`). It does not contain the off-diagonal
     correlation entries.
   - Those entries are the ones that reach `1e4` on the four bare starts
     (`correlation_0_01_real`, `correlation_1_10_real`,
     `correlation_0_10_imag`, `correlation_0_01_imag`).
   - So the energy level set is not compact. The energy, the only conserved
     quantity used here, does not confine those entries (`R §6`).
   - Past the event the largest coordinate goes from `1e4` to `1e10` in
     `0.25` to `0.33` time units on three of the four starts, the same at
     `rtol 1e-10` and `1e-12`. A finite-time singularity is not
     established (`R §6`).
6. **The growing direction is not the relative-mode phonon population.**
   Over the bounded phase (`t = 70` to `425.58`, until the largest
   coordinate first exceeds 5), the leading forward Lyapunov vector along
   the `t = 60` trajectory puts these shares of its squared norm on each
   block (`R §0` item 4, `§2.2`):

   | block | share |
   |---|---:|
   | anomalous phonon covariance | `0.355` |
   | electron--phonon correlation | `0.333` |
   | phonon covariance | `0.147` |
   | of which the relative-mode population | `0.045` |

## 2. The cheap repair

The repaired chain has a fixed configuration:

- 27-key fixed dictionary, Hartree--Fock preparation with the exact-moment
  lift;
- Dykstra at 120 iterations onto both cones, every SSPRK(3,3) stage, at
  `h = 0.0025`;
- the Eq. (22) hyperplane at the drive-power reference;
- lift re-selection every 16 steps.

Task 11 is the same configuration from the exact correlated preparation,
with the declared-completion lift and the retry ladder, run to `t = 20`.

1. **Accuracy over `[0, T]`** (`R §13.6`, `cheap_repair_accuracy.json`):

   | route | T = 4 | T = 8 | T = 20 | T = 60 |
   |---|---|---|---|---|
   | repaired chain, Hartree--Fock | 0.0070 / 0.0011 / 0.0108 | 0.0331 / 0.0011 / 0.0372 | 0.0736 / 0.0012 / 0.1537 | 0.1604 / 0.0012 / 0.2523 |
   | bare closure, Hartree--Fock | 0.0522 / 0.0421 / 0.1296 | 0.0823 / 0.0440 / 0.2937 | 0.1682 / 0.0452 / 0.3320 | event at `54.46` |
   | task 11, exact correlated | 0.0116 / 0.0040 / 0.0093 | 0.0281 / 0.0049 / 0.0330 | 0.0681 / 0.0053 / 0.0560 | not run |
   | bare closure, exact correlated | 0.0296 / 0.0276 / 0.1109 | 0.0448 / 0.0222 / 0.1382 | 0.1431 / 0.0182 / 0.1932 | 0.3054 / 0.0162 / 0.3248 |

2. **Cost, one Hartree--Fock trajectory, each wall with its session.**

   | wall | horizon | session and conditions | s per unit |
   |---|---|---|---:|
   | the recorded chain, all 14 segments | `t = 0` to 60, `12,990.087` s | its own run (`R §13.2` (e); `DI` §9 prints `12,990.1`) | `216.5` |
   | the same recorded run, its first five segments | `t = 0` to 20, `4,296.9` s | its own run (`R §13.2` (e); `DI` §4.1) | `214.8` |
   | `R §13.11` session, minimum of three runs (`138.7`, `138.8`, `139.2` s) | `t = 0` to `0.4` | load average `3.7` to `4.3` on 14 cores at the session's map-timing rounds | `346.8` |
   | `R §14.4` session, one run of `132.8` s | `t = 0` to `0.4` | load average `2.2` to `6.0` over the session, mostly `3.0` to `4.3` | `332.1` |

   - **The `138.7`-s run of the `R §13.11` session** spent `117.2` s on 10
     lift re-selections (8 recorded as failures), `19.0` s on 480 Dykstra
     repairs and `2.5` s on velocity evaluations
     (`cheap_repair_timing_run1/summary.json`). `R §13.11` prints `18.8` s
     for the repairs; that is the second run's figure, and the receipt is
     used here.
   - **The `R §14.4` run** spent `112.5` s on re-selection, `17.9` s on
     Dykstra repairs and `2.4` s on velocity evaluations.
3. **It does not remove the instability for the repaired chain's states.**
   The test hands the state to the bare closure: 64 members, perturbation
   size `1e-4`, span 1500. All 64 fire from every repaired-chain state
   tested (`R §13.1`). Means in units after the state, with bootstrap 95
   percent intervals (20,000 resamples):

   | state at | repaired chain | exact trajectory from Hartree--Fock | ratio, repaired to exact (unperturbed ratio) |
   |---|---|---|---|
   | `t = 20` | `588.7 [558.4, 622.0]` | `128.2 [127.1, 129.5]` | `4.59 [4.35, 4.85]` (`4.1`) |
   | `t = 40` | `482.6 [461.4, 506.2]` | `291.7 [275.5, 308.7]` | `1.65 [1.54, 1.78]` (`1.3`) |
   | `t = 60` | `410.0 [389.2, 433.9]` | `361.7 [351.0, 373.1]` | `1.13 [1.07, 1.21]` (`0.94`) |

   - **The repaired mean falls with the state's time,** by `106 [67, 146]`
     from `t = 20` to 40 and `73 [41, 104]` from 40 to 60. The unperturbed
     trend (`527.1`, `481.2`, `350.8`) rests on three states of one chain
     (`R §13.1` (d)).
   - **Integrator sensitivity.** At `rtol 1e-12`, `atol 1e-14`, single
     members from the repaired `t = 20` state move by up to `12.3` units,
     and the 8-member mean by `-2.2` (0.4 percent) (`R §13.1` (d)).
   - **At `t = 60` the unperturbed repaired state is not longer-lived** than
     the exact one: `350.78` against `373.90`.
   - **The exception.** Task 11's `t = 20` state has the same repair
     configuration, from the exact correlated preparation. Its unperturbed
     run has no event in 1500 units, and 41 of 64 members have none. The
     restricted mean to 1500 is `1409.7`, a lower bound (`R §13.1` (g),
     (h)).
4. **The repair does not change the bare closure's linearization. It
   changes where that linearization is taken** (`R §0` item 5, `§4.2`).
   - Over `t = 0` to 20 the leading real part of the Jacobian averages
     `0.177` along the repaired chain, `0.211` along the exact trajectory
     and `0.206` along the bare Hartree--Fock trajectory.
   - The difference is the growing direction and its rate:

     | | bare trajectory | repaired chain |
     |---|---|---|
     | where the growing direction sits | moves into the electron--phonon correlation (share `0.31` to `0.57` from `t = 12` to 30) | stays in the phonon blocks (correlation share `0.05` to `0.23` from `t = 8`) |
     | mean growth rate over `t = 10` to 20 | `0.11` per unit | `0.045` per unit |
     | linear amplification over `[0, 20]` | `47.9` | `26.4` (a factor `1.8` less) |

   - The decisive difference is nonlinear. The bare state's largest
     coordinate reaches 5 at `t = 40.77` and `1e4` at `54.46`. The repaired
     state's is at most `2.1827` through `t = 60`.

## 3. The retained-Gram necessary condition as a predictor

The tested condition is that the retained 8-by-8 joint Gram of
`(1, b0, b1, b0^dag, b1^dag, sigma_x, sigma_y, sigma_z)`, with the shared
trace removed, is positive semidefinite (`run_unrepaired.py`). It is
measured by its minimum eigenvalue, "the Gram" (`R §13.3`).

- A correctly built negative moment Gram obstructs physical
  representability.
- Positivity of this retained submatrix, or of the selected minors, is only
  a necessary condition within the chosen truncation. It cannot certify a
  physical state or a positive 36-row extension.
- "Representability" below means only this tested necessary condition.

1. **The tested condition fails within units, from every tested state.**
   - Under the bare closure the Gram is already negative at the start state,
     or turns negative within `0.5` to `12` units of it, in all 19 escape
     ensembles.
   - The exact trajectory's states start at `+3.8e-3`, `+7.5e-3` and
     `+9.9e-3`, and cross 0 within `1.5`, `0.75` and `1.2` units.
   - The Hartree--Fock preparation has a Gram of rank 4 of 8 at `t = 0`,
     and is below `-1e-8` by `t = 0.015`.
   - The repaired chain's own `t = 20`, 40 and 60 states already have Gram
     `-1.65e-4`, `-1.90e-4` and `-2.65e-6`. Along the chain, sampled every
     `0.25`, the Gram is below 0 at 87 percent of 241 samples and below
     `-1e-4` at 29 percent, with minimum `-7.96e-4` at `t = 51.75`.
     Section 5 samples every `0.0025` instead and gets 24.2 percent and
     `-8.5e-4`.
2. **"Below `-1`" is a depth threshold, not the loss.** The first time
   below `-1` comes a median number of units before the event:
   - `175` to `318` units in the five tabled ensembles (member minimum 50);
   - `14.5` to `51` units in the bare Hartree--Fock, task-11 `t = 10` and
     task-13 `t >= 10` ensembles;
   - `14.5` to `724.2` units across all 19 ensembles.
3. **In this ensemble it does not discriminate escape.** In the task-11
   `t = 20` ensemble, 38 of the 41 members with no event in 1500 units cross
   Gram `< -1`, at a median `542.8` units. The 23 members that fire cross at
   the same median `542.8`. This is a finite, in-sample association, with 41
   of 64 members censored at 1500. Equal medians do not prove that the
   condition carries no information about escape.
4. **Within one ensemble it orders the events weakly or not at all**
   (finite, in-sample).
   - In each of the five tabled ensembles (64 members each), the first time
     below `-1` has rank correlation `-0.04` to `0.25` with the event time
     (`p = 0.047` to `0.95`).
   - In two of them the crossings fall within one to four `0.05`-unit
     samples, so the correlation is undefined in substance.
   - `R §13.3` prints the range as `-0.01` to `0.25`. The `R §10` table and
     the receipts (`escape2_analysis.json`, the refuter's `f_analysis.json`)
     give `-0.0443` for the repaired `t = 40` ensemble, so `-0.04` is used.
   - Across the 19 starting states the medians correlate `0.905`
     (`p = 1.0e-7`). That is partly by construction, because every crossing
     precedes its event.
   - The early crossings order the events weakly or not at all (rank
     correlation `-0.06` to `0.42` within an ensemble). The `0.42`
     (repaired `t = 60`) and `0.31` (exact `t = 60`) are distinguishable
     from zero (`p = 5e-4` and `0.013`).
5. **A tuned lead.** The thresholds here were tuned on the same five
   ensembles of section 10's table (of 19). That table itself covers 9 of
   the 23 stored thresholds. The earliest crossing that orders the members'
   events (rank correlation at least `0.9` over 64 members) comes a median
   number of units before the event:

   | quantity | thresholds | lead |
   |---|---|---|
   | the Gram | `-2.75` to `-4.5` | `18` to `38` units |
   | the largest coordinate | `2.4` to `4` | `12` to `26` units |

   - At equal predictive skill (`0.8`) the Gram is `1.2` to `6.2` units
     earlier. `R §13.3` prints `1.3`; the receipt `d_threshold_scan.json`
     gives `1.228` at the low end, so `1.2` is used.
   - At section 10's own fixed thresholds, Gram `< -5` leads the coordinate
     passing 4 in five of five ensembles, by `1.7` to `6.5` units.
   - Neither leads by more than about 40 units, which is under 10 percent
     of the `281` to `577` units to the event.

## 4. The guard: the second-order clip, with and without the energy held

The exact constraint set and the capped algorithm that targets it are
separate things, and are kept separate below.

**What the clip targets** after every RK4 step (`R §13.4`). These are held
only to the recorded residual:

- without the energy hold, a bound violation of at most `3.6e-12`;
- with it, `1e-12`, except at the non-converged batch steps of item 2;
- the `maps2` and `run_spec.py` runs (items 3, 6 and 7, and section 5)
  record no bound residual at all.

The targeted families are:

- the Bloch ball, so the electronic `2 x 2` density is positive
  semidefinite;
- the phonon covariance positive semidefinite (`N_qq >= 0` and
  `|N_01|^2 <= N_00 N_11`);
- `|A_qq|^2 <= N_qq (N_qq + 1)` and the `|A_01|` bound;
- six correlation Cauchy--Schwarz bounds along the Pauli `x`, `y` and `z`
  axes;
- `Tr C[q] = 0`.

**What it does not hold.** These figures are minima and fractions over
unit records over `t <= 60`, one unperturbed trajectory (receipts `task2_corrections/obs_accuracy_*.json`):

- the retained joint Gram;

  | variant | Hartree--Fock: fraction below `-1e-4` (minimum) | exact correlated: fraction below `-1e-4` (minimum) |
  |---|---|---|
  | the clip | 64 percent (`-0.366`) | 90 percent (`-0.627`) |
  | the clip with the coherent-field restoration | 69 percent (`-0.366`) | 98 percent (`-0.633`) |
  | the clip with the energy hold | 74 percent, 75 at `h/2` (`-0.366`) | 97 percent (`-0.625`) |

- the boson moment matrix `[[N^T, A^*], [A, 1 + N]]`. Its minimum is
  `-0.142` from Hartree--Fock and `-0.302` from exact correlated under the
  clip alone, and `-0.142` and `-0.233` with the energy hold;
- the Eq. (22) energy, unless the energy hold is added.

The map acts at every step: the fraction of member-steps clipped is `1.0`.

**The energy hold** (`R §13.6`). The map alternates the clip with the exact
root of `E(x + s grad E) = E_target`. `E_target` is the computed energy
after the RK4 step, before the map. It is not taken from the exact
reference, and it is not an independently accumulated drive-work
reference. The map itself does no work, but by changing the state it can
change the work the drive does later. The
energy residual is at most `1.0e-12` in every run. The bounds are
alternated to `1e-12`, at most 200 alternations. Where that cap is reached
the step is counted as not converged (item 2).

1. **Compactness, and its two conditions** (`R §13.2` (c)). Take an Eq. (22)
   energy fixed or bounded above, `E <= E_max`, together with `N >= 0`,
   `|r| <= 1`, the twelve 2-by-2 Gram minors and `Tr C[q] = 0`. That set
   is then compact. A trajectory stays in it only while every condition
   actually holds, which the capped map does not guarantee (item 2), and
   sampled extrema cannot show that the conditions hold at every point of
   an interval. The bound is sharper than section 11's; its global
   optimality is not shown:

   `E >= -2t + omega |B|^2 - 4 g |B| + omega Tr N - 2 g (sqrt N_00 + sqrt N_11)`.

   - That bound gives
     `|B| <= [4g + sqrt(16 g^2 + 4 omega (E_max + 2t + 2 g^2/omega))] / (2 omega)`
     and
     `sqrt N_qq <= [2g + sqrt(4 g^2 + 4 omega (E_max + 2t + 5 g^2/omega))] / (2 omega)`.
     The other blocks are bounded through these.
   - At `E_max = -3.5` the bounds are `|B| <= 4.90`, `N_qq <= 13.5`,
     `|A| <= 13.99` and correlations `<= 3.67`.
   - The `1e4` event needs `E >= 4.87e3`.
   - **This certificate needs trace control.** The retained Gram subtracts
     `Tr C[q]`. Take `N_00 = N_11 = n`, `x17 = -omega n / (2 g)` (about
     `-0.408 n` here) and all else zero. The energy `2 omega n + 4 g x17` is
     then exactly 0, and the Gram minimum is `+1`, yet `max|x| = n`. So the
     Gram held with the energy alone is not compact. This does not show that
     an exactly zero trace is the only control that would restore
     compactness.
   - **The energy must stay bounded above.** Here it is held exactly. The
     clip alone does not keep it bounded. Over 3000 units, 0 of 16 members
     fire, but the energy keeps rising from `t0 + 10`:

     | start | median energy change at 3000 | largest energy change at 3000 | largest recorded sampled value of the largest coordinate over 3000 units (records every 400 steps) | median largest coordinate at 3000 |
     |---|---:|---:|---:|---:|
     | exact correlated preparation | `+66.4` | `+319` | `285` | `73` |
     | exact `t = 60` state | `+3.89` | `+95.4` | `124` | `2.4` |

     Some members run away between 1200 and 1800 units (`R §13.9`). The
     recorded heating has not levelled off by 3000, but that does not show
     heating without bound.
   - **What is analytical and what is measured.** Compactness is the
     analytical bound above. Measured confinement with the energy hold stops
     at 600 units (section 8, item 1).
2. **A finite ablation of the families, with the capped map.** 8 members
   per cell, perturbation size `1e-4`, 600 units, `1e4` threshold, from both
   `t = 0` preparations (`R §13.5`). These outcomes are those of the tested
   capped algorithm, row by row. They are not general necessity or
   insufficiency results, least of all in rows where enforcement failed.
   - **With the exact-energy hold,** dropping any single family gives 0
     fired, with a sampled largest coordinate of at most `3.80`. In this
     finite test no single family is needed to avoid the event over 600
     units.
   - **Keeping only the phonon bounds.** Keeping only the trace pin, only
     `N_qq >= 0`, both, or `N_qq >= 0` with the anomalous bounds lets 8 of 8
     cross the threshold from both preparations. Most of those cells carry
     non-converged batch steps (table below). Only the Hartree--Fock
     `trace`, `n_diag` cell and the exact-correlated `trace` cell converged
     at every step. A capped map that leaves kept constraints unconverged
     cannot establish that the exactly enforced intersection is
     insufficient.
   - Keeping the trace pin, `N_qq >= 0` and the anomalous bounds together
     gives no event from Hartree--Fock (0 fired) but 8 from exact correlated
     in the 600-unit window.
   - Keeping only the correlation bounds gives 0 fired, but the largest
     coordinate reaches `169` at 600 from Hartree--Fock and is still
     growing.
   - **With the energy not held** (one application per step, residual at
     most `3.6e-12`), dropping the correlation bounds lets 7 of 8
     exact-correlated members cross the threshold (`502` to `576` units).
     Hartree--Fock members reach a sampled `2.3e3`. That is a finite
     ablation result, not mathematical necessity.
   - Dropping the anomalous bounds raises the post-pulse heating to `+190`
     (exact correlated) and `+7.8` (Hartree--Fock).
   - From exact correlated, the Bloch-ball clip is the source of the
     heating: without it the energy moves `+0.040` from `t = 10` to 600,
     against `+5.57` with it.
   - **Non-converged steps.** Three Hartree--Fock drop-one cells with exact
     energy did not converge at every step. The counts are batch steps:
     `minors_family.py` counts one non-converged step whenever the whole
     batch fails to converge, and the number of failing members was not
     retained.

     | cell | non-converged batch steps | largest remaining bound violation |
     |---|---:|---:|
     | drop `a` | 120 | `6.1e-3` |
     | drop `n_diag` | 10 | `5.8e-4` |
     | drop `trace` | 7 | `5.3e-8` |
     | exact `t = 60`, all families | 305 | `1.2e-7` |

     Keep-only cells, as non-converged batch steps / largest energy residual
     after the map (`family_*.json`; the largest recorded bound violation is
     at most `1e-12`, except `2.1e-10` in the two Hartree--Fock `corr` cells):

     | families kept | Hartree--Fock | exact correlated |
     |---|---|---|
     | `trace` | 1964 / `1.5e-11` | 0 / `9.1e-13` |
     | `n_diag` | 9 / `1.4e-12` | 846 / `1.1e-11` |
     | `trace`, `n_diag` | 0 / `9.1e-13` | 2 / `1.4e-12` |
     | `n_diag`, `a` | 3487 / `2.2e-11` | 4583 / `2.2e-11` |
     | `trace`, `n_diag`, `a` | 0 / `4.0e-15` | 26 / `3.6e-12` |
     | `corr` | 2 / `1.0e-12` | 0 / `9.9e-13` |
     | `corr`, `electronic` | 2 / `1.0e-12` | 1 / `1.0e-12` |

     Seven of the nine fired keep-only cells carry non-converged batch
     steps, up to `4,583`. The batched exact-correlated drop-`trace` cell
     recorded only the energy residual (section 8, item 13).
3. **A triggered guard** (`R §13.8`). The exact-energy clip is applied only
   where the largest coordinate exceeds 3 or the Gram falls below `-2.75`.
   - It prevents every event in 600 units: 0 of 16 from the repaired chain's
     `t = 60` state, 0 of 16 from the exact trajectory's.
   - The same 32 starts all fire under the bare closure, at `307` to `566`
     units.
   - It acts at `8.7e-4` and `9.7e-3` of member-steps.
   - **Its thresholds were tuned on the same ensembles** (`R §13.3`). No
     held-out test exists.
   - Before its first trigger it is the bare closure, so its agreement with
     bare dynamics is not an independent accuracy benefit.
   - Its zero events are finite, censored observations of a capped
     algorithm, not a certificate of exact constraint preservation.
     `run_spec.py` records no alternation count for the 16-member
     ensembles.
   - **From the Hartree--Fock preparation** (one unperturbed trajectory) it
     acts at a single step, `t = 37.26`, the first time the Gram falls below
     `-2.75`. The largest coordinate stays at most `2.56`. At that step the
     alternation reaches its 200-iteration cap, with the energy residual 0
     and the bounds not converged to `1e-12`. That one map moves the state
     off the bare trajectory, which fires at `54.46`. No event follows to
     `t = 60`, and the `T = 60` all-31 RMS is `0.429`, against `0.346` to
     `0.367` for the always-on clip maps.
4. **Cost.** One trajectory per preparation, `t = 0` to 20, `h = 0.0025`,
   minimum of three interleaved rounds:

   | session | bare RK4 step | clip | clip with the energy hold |
   |---|---|---|---|
   | `R §13.11` (load `3.7` to `4.3`, 14 cores), Hartree--Fock | `0.140` s per unit | `0.165` (`1.18` times bare) | `0.216` (`1.55` times) |
   | `R §13.11`, exact correlated | `0.139` | `0.164` (`1.18` times) | `0.426` (`3.07` times) |
   | `R §14.4` (load `2.2` to `6.0`), Hartree--Fock | `0.130` | not measured | `0.203` |
   | `R §14.4`, exact correlated | `0.132` | not measured | `0.408` |

   The clip is about 21 square roots and 22 comparisons per state per step.
5. **Observed first-order behaviour in `h`** (`R §13.4`). With a map after
   every RK4 step, the ratio of successive step-halving differences is
   `1.91` to `2.39` at `t = 20`, against `16.0` for the bare closure. This
   holds for the tested always-on variants on one trajectory per
   preparation. It is not an order theorem or a continuum-accuracy
   statement. Per-variant score changes
   between `h` and `h/2` are the last column of the tables in item 6.
6. **Accuracy: the guard does not approach the cheap repair on the
   observables** (`R §13.6`; one unperturbed trajectory per row, scores
   over `[0, T]`). The last column is the largest change of any of the
   twelve scores between `h` and `h/2`. The guard can improve a selected
   metric over the bare closure while failing to match the cheap repair on
   others, so the three metrics are compared separately below, by
   preparation, horizon and refinement.

   **Hartree--Fock preparation**

   | map | T = 4 | T = 8 | T = 20 | T = 60 | event (elapsed) | largest change, `h` to `h/2` |
   |---|---|---|---|---|---:|---:|
   | bare closure | 0.0522 / 0.0421 / 0.1296 | 0.0823 / 0.0440 / 0.2937 | 0.1682 / 0.0452 / 0.3320 | event | 54.46 | 2.1e-11 |
   | clip | 0.0356 / 0.0436 / 0.1313 | 0.0659 / 0.0466 / 0.2969 | 0.1654 / 0.0480 / 0.3145 | 0.2715 / 0.0858 / 0.3576 | none to 60 | 1.9e-3 |
   | clip with the coherent-field restoration | 0.0358 / 0.0425 / 0.1310 | 0.0660 / 0.0456 / 0.2960 | 0.1654 / 0.0473 / 0.3136 | 0.2437 / 0.0353 / 0.3458 | none to 60 | 4.1e-4 |
   | clip with the energy hold | 0.0356 / 0.0437 / 0.1313 | 0.0659 / 0.0469 / 0.2970 | 0.1655 / 0.0487 / 0.3146 | 0.2890 / 0.0495 / 0.3668 | none to 60 | 4.2e-4 |
   | Gram cone (one eigenvalue clip per step) | 0.0359 / 0.2292 / 0.1158 | 0.0470 / 0.3274 / 0.2670 | 0.0859 / 0.3841 / 0.2584 | 0.1979 / 0.4902 / 0.3048 | none to 60 | 5.3e-4 |
   | Gram cone, then one exact energy root | 0.0464 / 0.0358 / 0.1085 | 0.0673 / 0.0461 / 0.2323 | 0.0839 / 0.0512 / 0.4595 | 0.1285 / 0.0534 / 0.5479 | none to 60 | 1.3e-2 |
   | boson moment matrix plus clip, energy held | 0.0448 / 0.0409 / 0.1057 | 0.0621 / 0.0428 / 0.2478 | 0.1611 / 0.0439 / 0.2651 | 0.2609 / 0.0443 / 0.3042 | none to 60 | 9.4e-5 |
   | triggered guard | 0.0522 / 0.0421 / 0.1296 | 0.0823 / 0.0440 / 0.2937 | 0.1682 / 0.0452 / 0.3320 | 0.3155 / 0.0457 / 0.4294 | none to 60 | 4.6e-11 |
   | cheap repair (repaired chain) | 0.0070 / 0.0011 / 0.0108 | 0.0331 / 0.0011 / 0.0372 | 0.0736 / 0.0012 / 0.1537 | 0.1604 / 0.0012 / 0.2523 | none | --- |

   **Exact correlated preparation**

   | map | T = 4 | T = 8 | T = 20 | T = 60 | event (elapsed) | largest change, `h` to `h/2` |
   |---|---|---|---|---|---:|---:|
   | bare closure | 0.0296 / 0.0276 / 0.1109 | 0.0448 / 0.0222 / 0.1382 | 0.1431 / 0.0182 / 0.1932 | 0.3054 / 0.0162 / 0.3248 | none to 60 | 2.8e-10 |
   | clip | 0.0269 / 0.1008 / 0.1081 | 0.0506 / 0.1096 / 0.1214 | 0.1586 / 0.1603 / 0.1686 | 0.2810 / 0.3070 / 0.2559 | none to 60 | 1.3e-3 |
   | clip with the coherent-field restoration | 0.0362 / 0.0793 / 0.1299 | 0.0459 / 0.0877 / 0.1336 | 0.1394 / 0.0924 / 0.1578 | 0.2426 / 0.0992 / 0.2122 | none to 60 | 3.1e-2 |
   | clip with the energy hold | 0.0192 / 0.0352 / 0.1059 | 0.0465 / 0.0353 / 0.1148 | 0.1347 / 0.0353 / 0.1537 | 0.3004 / 0.0354 / 0.2513 | none to 60 | 1.6e-4 |
   | Gram cone | 0.0218 / 0.4429 / 0.1225 | 0.0521 / 0.6315 / 0.1326 | 0.2036 / 0.9321 / 0.2229 | 0.2105 / 1.0847 / 0.2231 | none to 60 | 1.2e-3 |
   | Gram cone, then one exact energy root | 0.0991 / 0.1367 / 0.4508 | 0.1525 / 0.1313 / 0.4890 | 0.2222 / 0.1280 / 0.5257 | 0.2153 / 0.1265 / 0.5279 | none to 60 | 4.1e-2 |
   | boson moment matrix plus clip, energy held | 0.0192 / 0.0352 / 0.1059 | 0.0497 / 0.0353 / 0.1155 | 0.1856 / 0.0353 / 0.1931 | 0.3024 / 0.0354 / 0.2702 | none to 60 | 4.8e-4 |
   | cheap repair (task 11) | 0.0116 / 0.0040 / 0.0093 | 0.0281 / 0.0049 / 0.0330 | 0.0681 / 0.0053 / 0.0560 | not run | none | --- |

   - **Against the cheap repair, on the energy.** No map comes within `6.6`
     times the cheap repair's Eq. (22) RMS at any horizon. The best map is
     `29` to `38` times worse from Hartree--Fock and `6.6` to `8.7` times
     from exact correlated.
   - **Against the cheap repair, on the site-0 occupation.** The repair is
     better at `T = 4` and 8 from both preparations, and at `T = 20` from
     exact correlated. From Hartree--Fock at `T = 20` the Gram-cone maps
     come within `1.1` times (`0.0839` against `0.0736`). At `T = 60` the
     Gram cone with one exact energy root is lower (`0.1285` against
     `0.1604`), but it is not `h`-converged.
   - **Against the bare closure, a gain.** The clip lowers the site-0 RMS
     at `T = 4` (`0.0356` against `0.0522` from Hartree--Fock, `0.0269`
     against `0.0296` from exact correlated). It stays within `0.003` at
     `T = 20` from Hartree--Fock.
   - **Against the bare closure, a loss.** From exact correlated the clip
     raises the Eq. (22) RMS over `[0, 20]` from `0.018` to `0.160`; the
     energy hold brings it to `0.035`.
   - **The remaining energy error.** What is left of the Eq. (22) RMS under
     the hold (`0.035` exact correlated, `0.044` to `0.050` Hartree--Fock)
     is consistent with a difference between the work the drive does on the
     closure and on the exact state. At `t = 1` the closure's energy is
     `-3.567` with or without a map, against the exact `-3.597`.
     - That rounded agreement at one time does not establish equal work
       histories.
     - An energy-neutral map can change later drive work by changing the
       state.
     - So the attribution of the whole RMS difference to pulse work is not
       proved.
   - **The critic's assessment, verbatim** (`R §13.4`, from the
     completeness critic of 2026-10-11): *the minors clip is a candidate
     cheap divergence guard, not a representability repair, and it does not
     improve accuracy.* Its comparison is with the cheap repair. Against the
     bare closure some errors improve and others worsen, as the two bullets
     above show.
7. **The Gram-cone and boson moment maps** (`R §13.7`; 8 members, 600
   units, from both `t = 0` preparations; minima over unit records). None
   has an event in 600 units (0 of 8 per preparation). Each is a one-pass
   clip and map-back, or such a pass followed by an energy root, so none is
   a projection onto a fully enforced intersection. Their negative Gram and
   boson eigenvalues below quantify that.

   | map | heating | retained Gram minimum | below `-1e-4` | boson matrix minimum |
   |---|---|---|---|---|
   | Gram cone | `+0.77` (Hartree--Fock), `+1.81` (exact correlated) | `-6.2e-4`, `-1.4e-3` | | |
   | Gram cone, then one exact energy root | energy held | `-1.76`, `-2.20` | `99.7`, `99.8` percent of unit records | |
   | boson moment matrix plus clip, energy held | energy held | `-0.60`, `-0.73` | | `-1.8e-3`, `-1.4e-3` |

8. **Starts satisfying the tested necessary constraints escape as before**
   (`R §13.10`). The repaired
   chain's and the exact trajectory's `t = 20` states were perturbed (16
   members, sizes `1e-4` and `1e-3`), carried into the clip's set or the
   Gram cone, then handed to the bare closure. All 128 fire:

   | start | mean time to the event | section-9 mean, unprojected (64 members) |
   |---|---|---:|
   | repaired chain | `420` to `541` | `588.7` |
   | exact trajectory | `125.8` and `130.3` | `128.2` |

   The projected ensembles have 16 members against section 9's 64, and the
   exact stored states and the perturbed moment vectors have different
   provenance. So the means do not isolate a projection effect. In these
   tests, satisfying the tested necessary constraints at the start did not
   prevent escape. That is not a proof that escape is independent of
   starting-state geometry.

## 5. The hybrid: the guard every step, the repair every `Delta` units

**Definition** (`R §14.1`). The bare closure runs with the exact-energy clip
after every RK4 step at `h = 0.0025`. At `t = Delta, 2 Delta, ..., 60` one
recorded repair step is applied:

1. the 29 raw moments are taken from the closure state;
2. the 27 dictionary moments the repair needs, and the closure does not
   carry, are set by the zero-cumulant rule;
3. the lift is re-selected by the recorded rule;
4. one Dykstra-120 repair is applied onto both cones, with the Eq. (22)
   hyperplane at the energy carried in;
5. the result is mapped back, and the clip applied once more.

**The zero-cumulant rule is an assumption of this lane.** It equals the
exact-moment dictionary of the Hartree--Fock product state to `4.4e-16`. It
differs from the exact correlated ground state's dictionary by up to
`0.378`.

**The table columns.**

- Accuracy is over `[0, T]`.
- "s per unit" is the arm's own wall over `t = 0` to 60 in one serial
  session (`R §14.4`), which ran with load average `2.2` to `6.0`.
- The Gram columns are minima and fractions over all 24,001 grid samples in
  `[0, 60]`, with the post-repair state at repair times.
  - This sampling differs from section 4's unit records. That is why the
    guard alone shows `-0.381` and `75.0` percent here (Hartree--Fock),
    against `-0.366` and 74 percent in section 4.
- The escape column comes from the arm's `t = 60` state under the bare
  closure: 8 members, size `1e-4`, span 600. It gives the number fired by
  600, the restricted mean to 600 (a lower bound when members are
  censored) and the unperturbed state's own event.

**Hartree--Fock** (in the same session: bare step `0.130` s per unit, clip
with the energy hold `0.203`):

| `Delta` | T = 4 | T = 20 | T = 60 | s per unit | Gram minimum; fraction below `-1e-4` | escape |
|---|---|---|---|---:|---|---|
| 1 | 0.0318 / 0.0277 / 0.1364 | 0.1420 / 0.0288 / 0.2695 | 0.2506 / 0.0289 / 0.2919 | 3.258 | -0.200; 0.074 | 0; 600.0; none |
| 2 | 0.0370 / 0.0432 / 0.1392 | 0.1528 / 0.0477 / 0.2906 | 0.2600 / 0.0484 / 0.3063 | 1.711 | -0.354; 0.216 | 0; 600.0; none |
| 5 | 0.0356 / 0.0437 / 0.1313 | 0.1603 / 0.0487 / 0.2966 | 0.2666 / 0.0495 / 0.3211 | 0.778 | -0.381; 0.400 | 0; 600.0; none |
| 10 | 0.0356 / 0.0437 / 0.1313 | 0.1638 / 0.0487 / 0.3020 | 0.2782 / 0.0495 / 0.3335 | 0.480 | -0.381; 0.809 | 8; 481.4; none |
| 20 | 0.0356 / 0.0437 / 0.1313 | 0.1655 / 0.0487 / 0.3146 | 0.2852 / 0.0495 / 0.3570 | 0.357 | -0.381; 0.863 | 8; 227.9; 223.0 |
| infinity (guard alone) | 0.0356 / 0.0437 / 0.1313 | 0.1655 / 0.0487 / 0.3146 | 0.2890 / 0.0495 / 0.3668 | 0.224 | -0.381; 0.750 | 8; 368.8; 405.4 |
| every stage (repaired chain) | 0.0070 / 0.0011 / 0.0108 | 0.0736 / 0.0012 / 0.1537 | 0.1604 / 0.0012 / 0.2523 | 332.1 (`R §14.4` session, `t = 0` to `0.4`, one run); 216.5 (recorded, `t = 0` to 60, another session) | -0.00085; 0.242 | 8; 373.6; 350.8 |

**Exact correlated** (in the same session: bare step `0.132`, clip with the
energy hold `0.408`):

| `Delta` | T = 4 | T = 20 | T = 60 | s per unit | Gram minimum; fraction below `-1e-4` | escape |
|---|---|---|---|---:|---|---|
| 1 | 0.0225 / 0.0532 / 0.1289 | 0.0927 / 0.0599 / 0.1579 | 0.3788 / 0.0610 / 0.3281 | 3.081 | -0.197; 0.609 | 3; 584.6; none |
| 2 | 0.0280 / 0.0377 / 0.0900 | 0.0918 / 0.0407 / 0.1612 | 0.3710 / 0.0412 / 0.3240 | 1.623 | -0.195; 0.889 | 4; 512.2; 585.9 |
| 5 | 0.0192 / 0.0352 / 0.1059 | 0.0575 / 0.0353 / 0.1047 | 0.2857 / 0.0354 / 0.2441 | 0.804 | -0.433; 0.414 | 0; 600.0; none |
| 10 | 0.0192 / 0.0352 / 0.1059 | 0.1200 / 0.0353 / 0.1463 | 0.2131 / 0.0354 / 0.1975 | 0.529 | -0.749; 0.997 | 0; 600.0; none |
| 20 | 0.0192 / 0.0352 / 0.1059 | 0.1347 / 0.0353 / 0.1537 | 0.2725 / 0.0354 / 0.2459 | 0.714 | -0.749; 0.986 | 0; 600.0; none |
| infinity (guard alone) | 0.0192 / 0.0352 / 0.1059 | 0.1347 / 0.0353 / 0.1537 | 0.3004 / 0.0354 / 0.2513 | 0.353 | -0.749; 0.974 | 6; 504.6; 478.5 |
| every stage (task 11, to `t = 20`) | 0.0116 / 0.0040 / 0.0093 | 0.0681 / 0.0053 / 0.0560 | not run | not measured | --- | no `t = 60` state |

1. **The energy gap.** The best arm (Hartree--Fock, `Delta = 1`) is `24`
   times worse than the repaired chain on the Eq. (22) energy over
   `[0, 60]` (`0.0289` against `0.0012`). The other Hartree--Fock arms
   (`0.0484` to `0.0495`) are 40 to 41 times worse (derived from the
   table).
   - Over `[0, 60]` the best arm is `1.56` times worse on the site-0
     occupation and `1.16` times on all 31.
   - Over `[0, 4]` it is `4.6` and `26` times worse on the occupation and
     the energy (receipts: `0.0318386 / 0.0069780` and
     `0.0276566 / 0.0010543`). `R §14.0` prints `4.5`, from the rounded
     table values.
2. **From exact correlated the effect of `Delta` is not monotone.**
   - At `T = 60`, `Delta = 1` and 2 are worse than the guard alone (site-0
     `0.3788` and `0.3710` against `0.3004`; all 31 `0.3281` and `0.3240`
     against `0.2513`).
   - `Delta = 5`, 10 and 20 are better on site-0 and all 31 at `T = 60`
     (`Delta = 10`: `0.2131 / 0.1975` against `0.3004 / 0.2513`).
     `Delta = 5` and 10 are also better at `T = 20`.
   - No arm lowers the Eq. (22) RMS below the guard's `0.0354`. Repairs
     during the pulse raise it over `[0, 60]` to `0.0610` at `Delta = 1` and
     `0.0412` at `Delta = 2`.
   - `R §14.0` item 2 says the sparse repair "does not improve on the guard"
     from this preparation. Its own table (`R §14.3`) contradicts it; this
     file follows the table.
   - The one cell below task 11 (`Delta = 5`, site-0 over `[0, 20]`:
     `0.0575` against `0.0681`) has neighbours at `0.0918` and `0.1200`.
3. **Over the guard, from Hartree--Fock.** The `T = 20` and `T = 60`
   scores improve steadily as `Delta` shrinks. `Delta = 1` lowers the
   `T = 60` scores by 13, 42 and 20 percent, at `14.5` times the guard's
   wall.
   - At `T = 4` the improvement is metric-dependent. `Delta = 1` improves
     the site-0 occupation (`0.0318` against `0.0356`) and the energy
     (`0.0277` against `0.0437`), but worsens all-31 RMS (`0.1364` against
     `0.1313`). `Delta = 2` slightly improves the energy (`0.0432`) and
     worsens the other two (`0.0370`, `0.1392`).
   - No tested arm improves all three `T = 4` scores.
4. **The lift re-selection costs most of the wall and changes nothing.** It
   returned "infeasible" in all 222 main-arm events and in the 12 events of
   the `h/2` arm, 234 in all, so the carried lift was always used.
   - Averaged over each arm's events, it takes `2.4` to `3.0` s of a mean
     event wall of `2.5` to `3.1` s. Single events take `2.2` to `3.6` s of
     `2.2` to `3.7` s (`arm_*.json` records).
   - The repair call itself takes `0.04` s.
   - The refused branch returns a copy of the carried lift. That is
     source-level support for equal repair inputs if the re-selection were
     skipped. Solver and RNG side effects were not audited, and no
     skipped-selection run checked bitwise equality.
   - With that caveat, the same repair inputs would cost `0.22` to `0.29` s
     per unit with the re-selection skipped (derived from the same run, not
     measured). The exception is exact correlated `Delta = 20` at `0.594`,
     whose guard alternation reached its 200-iteration cap; that is also
     why its wall (`0.714`) exceeds `Delta = 10`'s.
   - With the re-selection, the hybrid is 100 to 930 times cheaper than the
     cheap repair's `332.1` s per unit in the same session. That wall
     includes 8 of 10 failed lift re-selections. The ratio compares the
     measured protocols, not equally accurate, successfully accepted
     methods.
5. **The escape test does not order with `Delta`.** From Hartree--Fock, 0 of
   8 fire by 600 at `Delta = 1`, 2 and 5, while all 8 fire at `Delta = 10`
   and 20 and for the guard alone. From exact correlated the counts are 3,
   4, 0, 0, 0 and 6 of 8 for `Delta = 1`, 2, 5, 10, 20 and infinity.
6. **The repair does not reach its cone, and the Gram is not held.**
   - The Dykstra repair at 120 iterations never reaches the extension cone:
     the minimum after every event is `-1.85e-4` to `-4.1e-3`.
   - Every post-repair, post-guard retained-Gram minimum is negative
     (largest `-7.75e-6`, smallest `-0.196`). Only 99 of the 234 events
     leave it above its pre-event value.
   - The exact-correlated `Delta = 20` arm reaches the guard's
     200-iteration cap, and `maps2.run` keeps no bound-residual receipt for
     those steps.
   - These are results of a capped fallback hybrid, not of a successful
     cone repair.
7. **Step dependence.** The `h/2` check on Hartree--Fock `Delta = 5` moves
   the scores by at most `1.8e-4`. That is one tested refinement of one arm;
   it does not show that the table is converged. The other arms were not
   checked.
8. **What the receipts prove, and what they do not.**
   - All 13 arms reached `t = 60` with no resume.
   - All 117 escape integrations ended with a nonfailure solver status: 76
     reached 600 and 41 stopped at the terminal event.
   - The arm walls exclude preparation, checkpoints, the escape test,
     scoring and the report, so the 1200-second whole-increment gate is
     unestablished.
   - The 0-of-8 counts are censored at 600 units, the accuracy is one
     trajectory per arm, and the zero-cumulant dictionary is an
     assumption.

## 6. Reading for the paper

**Scope first.** The manuscript's regularization slot asks for the high-`U`
failure mode (tex line 350): density-matrix positivity loss, stiffness,
correlation blow-up, energy drift or self-consistency instability.

- The runs here are the Holstein dimer at `lambda_ep = 1.5`, `gamma = 0.5`,
  with no Hubbard `U`. That is the case the manuscript treats as the smoke
  case before the high-`U` stress tests (tex line 344).
- So what follows is the archive closure's failure at strong
  electron--phonon coupling. It is offered for the high-`U` slot only by
  analogy. **High `U` was not run.**

The reading uses the manuscript's vocabulary: the equal-time non-Markovian
equations of motion for `rho`, the phonon covariance, the anomalous phonon
covariance, `B` and `delta rho_12^q`.

- **What the archive closure does.** It is fast. One bare RK4 trajectory at
  `h = 0.0025` costs `0.139` to `0.140` s per time unit in the `R §13.11`
  timing session, and `0.130` to `0.132` in the `R §14.4` session (minimum
  of three, `t = 0` to 20, one trajectory). On these diagnostics its failure
  shows no energy drift and no stiffness at the tested step:
  - on each bare trajectory the total energy after the external pulse
    (Eq. (22)) is the same to six decimals at twelve sampled times through
    the `1e4` event. It was not recorded during the growth past the event;
  - the trace of the flow's Jacobian vanishes;
  - the one-step backward-Euler matrix at `dt = 0.0025` has condition
    number `1.008` to `1.038` wherever the largest coordinate is below 5
    (`R §0` item 6). That is the conditioning of one matrix at one step on
    sampled bounded states, not a general nonstiffness certificate.
- **What breaks: correlation blow-up.** The failure is growth of the
  off-diagonal entries of `delta rho_12^q`, which the energy does not
  contain. Past the `1e4` threshold they reach `1e10` within `0.25` to
  `0.33` time units on three of four starts. Mathematical unboundedness and
  a finite-time singularity are not established.
  - From the exact trajectory's `t = 60` state the growth is preceded by
    positive finite-time stretching, with a largest finite-time exponent of
    `0.0468` over `t = 60` to 400.
  - From the Hartree--Fock preparation the growth is reached directly in
    `54.46` units (mean exponent `0.31` over `t = 0` to 54).
- **Where it happens.** It happens from:
  - the Hartree--Fock preparation at `t = 54.46`;
  - the exact correlated preparation at `141.41`;
  - the repaired chain's and the exact trajectory's states at `t = 20`, 40
    and 60 (means `128` to `589` units after the state).

  The exception is task 11's `t = 20` state: its unperturbed run has no
  event in 1500 units, and 41 of 64 members have none.
- **Positivity.**
  - The retained joint Gram of the electron--boson second moments is
    already negative at the start, or turns negative within `0.5` to 12
    units, in all 19 escape ensembles. The exact trajectory's states cross 0
    within `0.75` to `1.5` units, and the Hartree--Fock preparation is below
    `-1e-8` by `t = 0.015`.
  - This is loss of a necessary condition for joint representability, not
    of the electronic density matrix alone. The electronic density matrix's own positivity drift was
    not timed. The bare closure takes its lowest eigenvalue to `-216.4`
    from Hartree--Fock at the sampled times, against `0` to `-2.2e-16`
    under the clip (`R §11`).
  - In the tested, partly censored ensembles the joint loss does not
    discriminate which states blow up.
- **The other options.** In the manuscript's list:
  - Energy drift is not observed at the sampled times, and stiffness is not
    indicated at the tested step on sampled bounded states. Neither is
    excluded in general.
  - Self-consistency instability is neither confirmed nor excluded. The
    archive closure as run (`closed_scalar_rhs`) is an explicit ODE with no
    self-consistent step.
  - Of the manuscript's invariants (tex line 347), `Tr rho` and the
    Hermiticity of `rho` hold by construction of the 31 closed coordinates.
    Solver-step rejection counts were not recorded.
- **What the clip targets cheaply.** The capped clip targets:
  - `rho` positive semidefinite;
  - the phonon covariance positive semidefinite;
  - the anomalous-covariance and electron--phonon Cauchy--Schwarz bounds;
  - the trace of `delta rho_12^q` at zero.

  With the energy bounded above and every bound held, these bounds define a
  compact set. That is an analytical, conditional bound. The capped map
  holds the bounds only to its recorded residuals, not at every batch step.
  Its measured outcomes (no `1e4` event in the tested ensembles) stop at
  600 units. In the
  `R §13.11` session this costs `1.55` (Hartree--Fock) to `3.07` (exact
  correlated) times the bare RK4 step. The clip alone costs `1.18` times
  and is not compact: it heats.

  This is confinement without accuracy. Over `[0, 20]` the clip with the
  energy hold changes the bare closure's RMS as follows (derived from the
  `R §13.6` table, one unperturbed trajectory each):

  | preparation | site-0 occupation RMS | Eq. (22) energy RMS |
  |---|---|---|
  | Hartree--Fock | `0.0027` lower | `0.0035` higher |
  | exact correlated | `0.0084` lower | `0.0171` higher (`0.0182` to `0.0353`) |
- **What accuracy costs.** The Dykstra repair (section 2 of this file) gets
  the site-0 occupation within `0.0070` and the total energy within
  `0.0011` RMS over `[0, 4]` from Hartree--Fock.
  - From Hartree--Fock it costs `214.8` to `346.8` s per unit (section 2's
    four walls), against `0.130` to `0.216` s per unit for the bare step and
    the clip with the energy hold in the two timing sessions. That is three
    orders of magnitude (`1.0e3` to `2.5e3` times, `R §13.11`).
  - It does not remove the instability. Every state of the repaired chain
    tested (`t = 20`, 40, 60) blows up once handed back to the closure. The
    exception, again, is task 11's `t = 20` state.
- **No measured combination bridges the two.** From Hartree--Fock the best
  hybrid arm (`Delta = 1`) is 24 times worse than the repaired chain on the
  Eq. (22) energy over `[0, 60]`, and the other arms are 40 to 41 times
  worse. From exact correlated:
  - `Delta = 1` and 2 end worse than the clip at `T = 60`;
  - `Delta = 5`, 10 and 20 lower the site-0 and all-31 RMS at equal energy
    RMS, with no monotone trend;
  - there is no every-stage reference at `T = 60`.
- **For the quantum encoding slot** (tex line 353). Its options are state
  dimension, stiffness, nonlinearity, measurement burden or invariant
  preservation. On these classical `L = 2` runs:
  - state dimension (31 coordinates) and stiffness are not the bottleneck;
  - the failure lies in invariant preservation (positivity of the joint
    moments) under a nonlinear flow, volume-preserving by the exact
    divergence-free identity (`QF`);
  - measurement burden is not addressed here. The quantum-integration
    assessment is in `quantum_integration_feasibility_20261010.md` and
    `quantum_integration_negative_result_20261011.md`, in this folder.

## 7. Claims withdrawn during verification, with their replacements

Each line gives the withdrawn wording, then the replacement and where it is
recorded.

1. **"Never, through `t = 140`"** (exact correlated preparation, bare
   closure; `DI` §2.2) → fires at `t = 141.4057` (`R §0`).
2. **"Repaired states escape at a rate about five times lower than the bare
   closure's own states"** (`R §8`) → it compared two preparations. Against
   the exact trajectory from the same preparation, the repaired states live
   `4.6`, `1.65` and `1.13` times longer at `t = 20`, 40 and 60
   (unperturbed `4.1`, `1.3`, `0.94`) (`R §13.0`, `§13.1` (c)).
3. **"The bare closure's own states take `136.0`, `104.8`, `82.7`"** → those
   are one deterministic trajectory's remaining lifetime `141.41 - t`, plus
   `+14.6`, `+3.4`, `+1.2` from the perturbation (`R §13.1` (a)).
4. **"Censored maximum-likelihood rates"** → inverse mean times,
   `k / exposure`. Censoring enters 1 of 19 rows (`R §13.1`).
5. **"Admissible" read as representable** → three linear equalities only.
   The bare base states already violate the electronic, phonon and joint
   cones, while the exact-reference states start inside the joint cone
   (`R §13.0` item 3, `§13.1` (b)).
6. **"Five of nineteen ensembles fixed to within a unit"** → four of
   nineteen have all 64 events within one unit (`R §13.1` (e)).
7. **The after-delay rate left blank under a spread rule** → given for every
   complete row (`R §13.1` (f)).
8. **Task 11 at `t = 20`, rate `2.55e-4`** → restricted mean to 1500 is
   `1409.7` (a lower bound), the inverse mean at most `7.09e-4`, and the
   Kaplan--Meier median above 1500 (`R §13.1` (g)).
9. **The earliest bare exact-correlated `t = 20` event, `113.43`** → a fresh
   seed fires at `110.5` (`R §13.1`).
10. **The section-9 contrast "repaired chain against states the repair held
    cones on"** → task 11 and the repaired chain carry the same repair
    configuration. They differ in preparation and extension backend
    (`R §13.1` (h)).
11. **"Task 13 hands on the shortest-lived states"** → this holds only among
    the three repair chains (task 13 at `66` to `74` units elapsed).
    **"Task 11 the longest-lived of all"** → replaced: the task-11 times are
    erratic (`413.5`, `199.2`, `1038.1`, and more than 1500 for 41 of 64
    members at `t = 5`, 10, 15, 20) (`R §13.2` (a)).
12. **"The off-diagonal Cauchy--Schwarz bound lowers the escape rate `1.0`
    to `7.6` times; the escape moves to the phonon blocks"** → replaced
    (`R §13.2` (b), `§13.4`). The bound:
    - delays escape `5.2` times from Hartree--Fock;
    - delays it at least `3.6` times from exact correlated (restricted
      mean);
    - shifts it later from exact `t = 60` (median `366` to `469`, log-rank
      `p = 4e-9`);
    - has no detectable effect from repaired `t = 60` (`p = 0.87`).

    The escape is not confined to the phonon blocks.
13. **"Surviving members' largest coordinate `1.41` to `3.60`"** → this
    applies to the clip maps only. Survivors of the off-diagonal bound are
    mid-runaway (`257` and `513`) (`R §13.2` (b)).
14. **The compactness bound with `4 sqrt 2 g`** → valid but loose; the
    sharper bound of section 4 supersedes it (its global optimality is not
    shown). **"The retained joint Gram is
    sufficient with the energy" and "each part is measured"** → withdrawn:
    the Gram with the energy alone is not compact without the trace pin,
    and necessity was measured only in `R §13.5` (`R §13.2` (c)).
15. **"No member escapes, with or without the energy restored," read as
    confinement evidence** → no run in `R §8` to `§12` held the energy
    exactly. The restoration through `B` fails whenever the map has raised
    the energy past `B`'s own minimum (largest energy change `40.99` in
    `verify_minors.json`). The zero escapes follow from the measured heating
    (`R §13.2` (c)).
16. **"Every map above confines" and "every map above confines without the
    extension"** → only the clip and the clip with the coherent-field
    restoration. The off-diagonal bound alone does not (`R §13.2` (d)).
17. **"The 36-row positive extension is not needed for confinement at all"**
    (`R §11.3`) → restricted to the clip maps (`R §13.2` (d)). Its role is
    listed as unmeasured (`R §13.12`).
18. **"The clip costs about twice the bare step"** (`0.46` s per unit for the
    clip against `0.20` for the bare step) → `1.18` times, in one session.
    The two walls were separate runs under different load (`R §13.2` (e),
    `§13.11`). The refuter's `1.08`, also from a separate session, is
    superseded too.
19. **"About 500 times the wall" and "about one five-hundredth of the cheap
    repair's `217` s"** (`R §11.2`, `§11.3`) → `470` to `1340` times,
    depending on the wall used (`R §13.2` (e)). Measured in one session,
    `1.0e3` to `2.5e3` (`R §13.11`).
20. **"A handful of scalar comparisons and about twenty square roots"** →
    about 21 square roots (22 with the restoration) and about 22
    comparisons per state per step (`R §13.2` (e)).
21. **Accuracy "at `t = 20`"** → cumulative RMS over `0 <= t <= 20`
    (`R §13.2` (f)).
22. **"128 member-runs"** → 64 distinct initial conditions, each run under
    two maps (`R §13.2` (h)).
23. **"Every second-order bound" and "positivity bounds on every block"** →
    the explicit list in section 4. The clip holds neither the joint Gram,
    nor the boson moment matrix, nor the energy (`R §13.4`).
24. **Pulse inputs `0.22` (Hartree--Fock) and `0.24` (exact correlated)** →
    swapped: `0.24` and `0.22` (`R §13.4`).
25. **The off-diagonal bound from exact `t = 60`, exponential rate ratio
    `1.29`** → not a valid summary. The replacement is median `366` to
    `469`, log-rank `p = 4e-9` (`R §13.4`).
26. **"Task 13 faster as it heats"** → from `t >= 10` its states escape at
    absolute `t = 83.8` to `86.2`, nearly fixed, earlier than the bare
    closure from the same preparation. The `t = 5` state escapes at a mean
    absolute `421.4` (`R §13.4`).
27. **"Loses joint representability long before its event"** → negative at
    the start, or within `0.5` to 12 units, in all 19 ensembles. "Below
    `-1`" is a depth threshold, not the loss (`R §13.3`).
28. **"Carries no information about when the event comes"** → within each
    of the five tabled ensembles, the rank correlation of the first
    Gram `< -1` crossing with the event is `-0.04` to `0.25`. `R §13.3`
    prints `-0.01`; the `R §10` table and the receipts give `-0.0443`.
    Across the 19 starting states the medians correlate `0.905`, partly by
    construction (`R §13.3`).
29. **"No earlier and no better than the coordinate magnitude passing 4"** →
    with thresholds tuned on the same ensembles, the earliest ordering
    crossing comes a median `18` to `38` units before the event for the
    retained Gram, and `12` to `26` for the magnitude (`R §13.3`).
30. **The early crossings are "unrelated"** → they order the events weakly
    or not at all (rank correlation `-0.06` to `0.42` within an ensemble;
    `0.42` and `0.31` distinguishable from zero) (`R §13.3`).
31. **"The loss predicts that the bare closure will escape"** → withdrawn.
    No contrast case exists (`R §13.3`).
32. **"Bare `t = 20` value `-1.63` at 200 before"** → that rests on 4 of 64
    members. The 64-member value at 100 before is `-0.99` (`R §13.3`).
33. **The `1.1e-16` margin agreement** → it holds on bounded states. On the
    bare Hartree--Fock trajectory it is `6e-16` relative (`R §13.3`).
34. **"From exact correlated the sparse repair does not improve on the
    guard"** (`R §14.0` item 2) → the effect of `Delta` is not monotone.
    `Delta = 5`, 10 and 20 beat the guard on site-0 and all 31 at `T = 60`
    (`R §14.3` table; section 5, item 2 here).

Withdrawn in revision 2 (2026-10-18), after reviews 028 to 031:

35. **`x17 = -0.408 n` gives exactly zero energy** → the exact witness is
    `x17 = -omega n / (2 g)` (section 4, item 1).
36. **Non-converged "member-steps"** → batch-step counts. The number of
    failing members was not retained (section 4, item 2).
37. **"Supremum" of the largest coordinate** → the largest recorded sampled
    value, with records every 400 steps (section 4).
38. **"The tight lower bound"** → the sharper bound; its global optimality
    is not shown (section 4, item 1).
39. **"The Gram is held only at the repair times"** (`R §14.0` item 6) → it
    is not held at the repair times either (section 5, item 6).
40. **"The `T = 4` scores do not improve"** (this file's 2026-10-11
    revision) → the improvement is metric-dependent, and no tested arm
    improves all three (section 5, item 3).
41. **"The trajectories are therefore exactly those of the same hybrid with
    the re-selection skipped"** (`R §14.4`) → source-level support for equal
    repair inputs only (section 5, item 4).
42. **"The hold does not touch that difference"** (`R §13.6`) → an
    energy-neutral map can change later pulse work, and the attribution is
    not proved (section 4, item 6).
43. **"Confines", "is not numerical", "no stiffness" and "unbounded
    growth" as categorical statements** → scoped to the tested capped
    algorithms, refinements, steps and windows (sections 1, 4 and 6).

## 8. What remains unmeasured, in priority order

Every compute figure below is an **estimate**, named with the measured wall
it comes from. None of these studies is selected.

- **The campaign bound.** Any selected study would have to fit the
  campaign's 1200-second bound for a complete selected increment:
  preparation, reference, scoring, report, review and closure. A
  per-process limit is not enough.
- **Long studies.** Studies estimated in hours would need separately
  bounded, resumable subincrements under the existing fail-on-error and
  checkpoint gates.

1. **Confinement by the clip with the energy hold beyond 600 units.** This
   decides whether the compact set is reached in practice. It would take 8
   members to 3000 units in 600-unit batches.
   - Measured: the 8-member, 600-unit cells with all families and exact
     energy took `158.5` to `272.1` s (`task2_corrections`
     `family_*_drop-none_energy-exact.json`).
   - Estimate: about 3 to 5 minutes per batch.
2. **What makes the cheap repair accurate.** The candidates are the
   dictionary dynamics, the retained-Gram target, the 36-row positive
   extension and the hyperplane. Each needs one every-stage chain with one
   ingredient removed, to `t = 20`.
   - Measured: `214.8` to `346.8` s per unit (section 2).
   - Estimate: about `1.2` to `1.9` hours per chain, as many separately
     bounded subincrements.
3. **The exact correlated every-stage reference to `t = 60`,** with the
   repaired chain's driver. It gives the hybrid table its missing reference.
   - The same measured walls, which are Hartree--Fock profiles.
   - Estimate, extrapolated from those profiles: about `3.6` to `5.8` hours,
     as many separately bounded subincrements.
4. **A converged map onto the intersection of the Gram cone and the energy
   surface.** The Gram cone with one exact energy root is one pass of each
   and is not `h`-converged. Alternating them to convergence was not run
   (`R §13.12`).
   - Measured: one pass costs `0.197` to `0.198` s per unit (`R §13.11`).
   - Estimate: the alternation multiplies that by its iteration count,
     which is unknown.
5. **The hybrid with another dictionary rule,** for example carrying the
   repair's own dictionary between events.
   - Measured: each arm took at most `3.3` minutes (`R §14.8`).
6. **The hybrid at `Delta` below 1, with the re-selection skipped and run
   directly.**
   - Derived from Hartree--Fock (`R §14.7`): about `0.29` s per unit at
     `Delta = 0.5` and `0.37` at `0.25`.
   - Estimate: under half a minute per arm to `t = 60`.
7. **A converged Dykstra repair inside the hybrid.** At 120 iterations a
   repair takes `0.04` s. The iteration count needed is unknown.
8. **A held-out test of the triggered guard's thresholds,** on starts not
   used to choose them.
   - Measured: the 16-member, 600-unit triggered runs took `183` and
     `369` s (receipts).
   - Estimate: 3 to 6 minutes per start.
9. **Escape statistics with more members and a longer span** for the
   hybrid's and the guard's hand-on states. This would replace counts of 0
   of 8, whose exact 95 percent upper bound is `0.31` per member.
   - Measured: the section-9 ensembles of 64 members to 1500 units took
     `28` to `444` s each with 10 processes (`escape2_*.json`).
10. **Accuracy over perturbed members,** rather than one unperturbed
    trajectory per map. The triggered guard's accuracy is scored from
    Hartree--Fock only.
    - Measured: one trajectory to `t = 60` costs `0.130` to `0.426` s per
      unit.
    - Estimate: seconds to half a minute per member per map.
11. **The repaired chain with the repair switched off at `t = 20`, scored
    for accuracy.**
    - Measured: the bare closure costs `0.130` to `0.140` s per unit
      (`R §14.4` and `R §13.11` sessions).
    - Estimate: seconds.
12. **Repaired states other than `t = 20`, 40 and 60 of one chain,** and
    other chains.
    - Measured: 28 to 444 s per 64-member ensemble, as in item 9.
13. **The count of non-converged map steps in the batched exact-correlated
    drop-`trace` cell,** where only the energy residual was recorded. It
    needs a rerun with a counter.
    - Measured: the cell took `1529` s in six 100-unit batches.
    - Estimate: about 25 minutes, in batches under 20 minutes.
14. **`h/2` for the hybrid arms other than Hartree--Fock `Delta = 5`,** and
    for the exact correlated preparation.
    - Measured: the Hartree--Fock `Delta = 5` `h/2` arm took `60.2` s (receipt `arm_hartree_fock_D5_h0.00125.json`), and
      the `Delta = 1` arm took 3.26 minutes at `h`.
    - Estimate: about 1 minute per arm at `Delta >= 5` and about 3.5
      minutes at `Delta = 1`.
15. **Immutable execution provenance.** `run_spec.py` restores checkpoints
    without validating saved source, input or settings hashes, and computes
    its state-artifact hash after execution (`R §13.12`). The final
    manifest proves artifact integrity, not fixed source, input and settings
    across resumes. Any architectural reliance needs pre-execution locks and
    resume validation.
16. **The binding holds.** These classical results do not repair the three
    current strict resume-parity failures. They do not change root 025, the
    cone increment-3 hold or root messages 014 to 017 and 019. No guard,
    hybrid or chain study above, no dictionary, guard or skipped-selection
    policy, and no default, tolerance or execution follows from this file.
