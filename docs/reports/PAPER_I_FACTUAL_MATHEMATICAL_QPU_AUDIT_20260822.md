# Paper I factual, mathematical, and QPU audit

Last updated: 2026-08-22  
Audit branch: codex/paper-i-factual-audit-20260822  
Reader-facing manuscript snapshot: ADAPT---Paper-I commit ee00c56  
Audit mode: evidence collection only; do not edit the manuscript from this audit.

## Purpose

This is the single status ledger for checking that Paper I is:

1. mathematically faithful to the intended RA algorithm;
2. factually faithful to the executed benchmark routes and recorded data;
3. representative of an implementation that could be executed on a real QPU;
4. explicit at the scientific level without reproducing incidental code structure;
5. internally consistent across equations, prose, pseudocode, figures, tables,
   captions, and resource claims.

The audit distinguishes a genuine scientific contradiction from an
implementation-level difference that is mathematically equivalent. A literal
code-to-prose mismatch is not, by itself, a manuscript defect. Every proposed
reader-facing correction remains an author decision; this file records the
evidence and the strongest defensible interpretation.

## Status convention

- [x] DONE: the audit question has been answered from identified evidence.
  A DONE item may conclude either that the manuscript is correct or that a
  correction is required.
- [ ] TODO: evidence or comparison is incomplete.
- Finding: the scientific conclusion established by the audit.
- Manuscript consequence: the narrow reader-facing issue, if any. This is
  advisory only; this audit never edits Paper I.
- Equivalence allowance: why an implementation detail may be omitted without
  making the mathematics false.

## Protected audit rules

- Use the active non-iCloud repository or a source-locked run overlay.
- Stream large JSON artifacts; never load GB-scale checkpoints into memory.
- Preserve exact-state and ED quantities as reporting diagnostics; never turn
  them into online QPU controller inputs.
- Separate selection-time cost weights, estimator-query counts, circuit
  compilations, classical optimizer calls, and wall-clock work.
- For beam, distinguish selected-lineage estimator work from aggregate work
  across all executed children.
- For every alleged mismatch, state whether it changes the selected operator,
  accepted ansatz, predicted energy decrease, quantum measurement set,
  reported resource, or only an implementation receipt.
- Do not mark an item DONE merely because prose appears plausible.
- Do not change Paper I, its figures, or its tables from this branch.

## Current confirmed findings

### Trust-radius evidence

- [x] **RHO-01 — An empirical upper bound of 0.25 does not hold across all
  current Paper-I trajectories.**
  - Evidence: live Bundle-5 cluster 9674089, process 11, S--S
    plateau-insertion checkpoint, streamed from the execute node on 2026-08-22.
  - The checkpoint serializes two copies of the accepted history. Its four
    distinct increasing updates are 0.0206239 → 0.0324406,
    0.0162703 → 0.437185, 0.351931 → 1.75038, and
    1.75038 → 18.9614. The observed range is currently
    0.0162703 ≤ ρ ≤ 18.9614.
  - Finding: ρ₀ = 0.25 is an initialization, not an upper bound. At least one
    trajectory already displayed through its recorded prefix expands far
    above it.
  - Manuscript consequence: the sentence introduced in commit ee00c56 is
    qualified as applying to “completed trajectories.” The completed archive
    scan below is consistent with that literal scope, but the sentence does
    not represent the current live evidence and will become false when the
    S--S plateau trajectory is promoted. Revisit it after the run population
    is frozen; do not substitute a live maximum into the manuscript now.

- [x] **RHO-02 — “No radius exceeded 0.25” and “the radius never expanded”
  are distinct claims.**
  - Evidence: live S--S always-insertion checkpoint, process 17.
  - Its three distinct increasing updates are
    0.00392244 → 0.00456696 → 0.00630887 → 0.0112761, while its observed
    maximum remains 0.25.
  - Finding: a trajectory may expand relative to its preceding contracted
    value without exceeding its initialization.
  - Equivalence allowance: the manuscript may report the update equation
    without enumerating every expansion and contraction, but any empirical
    range must be computed over the complete declared run population.

- [x] **RHO-03 — The initial quick expansion count for W--S plateau was a
  parsing error and is excluded.**
  - Cause: an AWK comparison initially treated scientific-notation fields as
    strings. The forced-numeric recount gives zero expansions and maximum 0.25.
  - Finding: only forced-numeric streamed scans are admissible for this audit.

- [ ] **RHO-04 — Close the trust-radius range over every reported route.**
  - Audit all completed Bundle-5 archives, all six live strong-sector
    checkpoints, and the four L = 3 optional-control trajectories.
  - Record per run: distinct accepted updates, expansions, contractions,
    unchanged updates, minimum, maximum, and whether multiple beam branches
    are included.
  - Determine whether a reader-facing range should cover all stored branch
    states, only accepted/reporting lineages, or only singleton HH routes.
  - Do not propose a replacement numeric range until this item is DONE.
  - Completed Bundle-5 archive scan, streamed from the twelve transferred
    checkpoints on 2026-08-22: every archive has maximum 0.25. The minima are
    listed here so the eventual population range can be reproduced:

    | policy | W--W | I--W | S--W | W--S | I--S | S--S |
    |---|---:|---:|---:|---:|---:|---:|
    | Append | 0.0484978 | 2.64403e-14 | 0.0276002 | 9.08606e-13 | 1.29409e-14 | 1.14378e-4 |
    | Plateau | 1.83162e-15 | 2.91245e-14 | 9.61495e-5 | live | live | live |
    | Always | 5.12113e-14 | 2.40541e-13 | 0.0242772 | live | live | live |

  - Current live strong-sector accepted histories, streamed from cluster
    9674089 on 2026-08-22:

    | policy | W--S range | I--S range | S--S range |
    |---|---:|---:|---:|
    | Plateau | 9.08606e-13--0.25 | 1.29409e-14--0.25 | 0.0162703--18.9614 |
    | Always | 4.33163e-13--0.25 | 1.134e-13--0.25 | 6.02053e-13--0.25 |

  - Current L = 3 optional-control accepted histories: RA-Append,
    batch, beam selected lineage, and ablation each has maximum 0.25; their
    minima are 7.42722e-13, 0.0346309, 5.13965e-14, and 1.14006e-14,
    respectively.
  - The checkpoint payloads serialize repeated copies of accepted histories.
    Minima and maxima are unaffected; event counts remain pending a
    distinct-update pass and beam branch-scope audit.

### Estimator accounting

- [x] **EST-01 — Candidate-gradient and candidate-self-metric reuse has a
  defined mathematical accounting rule.**
  - Existing evidence:
    chtc/paper_i_ra_adapt_repair_20260727/FINDINGS_ESTIMATOR_LEDGER_DUPLICATION_20260821.md.
  - Mathematical contract: a measured primitive reused by a later phase at the
    same source state is charged once, regardless of whether the current code
    recomputed it.
  - Candidate correction:
    N_H_outer(raw) + N_H_refit(raw) + N_grad(unique) + N_metric(unique).
  - Equivalence allowance: repeated implementation invocations do not justify
    repeated manuscript charges when a later phase consumes the identical
    mathematical observable at the identical state.

- [x] **EST-02 — The phase-by-phase candidate-primitive reuse correction is
  established for the Bundle-5 weak-sector population.**
  - Evidence: all nine weak-sector Bundle-5 ledgers were streamed with
    `recount_estimator_phase_reuse.py`; their raw prefix counts reproduce
    `kstar_tables.json`, and the component-selective recount removes only
    repeated gradient and metric primitive identities.
  - Phase 0 to Phase I reuses the candidate gradient. Phase I acquires the
    candidate self metric. When the same positioned singleton survives to
    Phase II, its gradient and self metric are already available; only its
    candidate curvature is new. In measured-residual Bundle 5, the repeated
    occurrences are recorded primarily under `gradient_surface` and
    `phase1_candidate_self_metric`.
  - The earlier Bundle-3 anomaly is resolved: the four extra metric repeats in
    rounds 27 and 34 have scope `phase3_scaffold_geometry`; the primitive-id
    rule removes them without inventing a scope-name exception.
  - The source overlays shipped with Bundle 5 contain the same relevant
    gradient-surface and Phase-I self-metric call sites as the audited branch
    copy. Therefore the recount applies to the executed route, not merely the
    current branch.
  - Finding: use primitive identity, source state, and positioned record—not
    implementation scope names—to decide reuse. Cross-position equivalence is
    audited separately under REC-04; no reuse across distinct primitive ids
    is assumed here.

- [ ] **EST-03 — Recount every displayed estimator value under one closed
  convention.**
  - HH Table I at k-star.
  - Optional-control resource matrices.
  - Noise Table II.
  - Beam selected lineage versus aggregate parallel execution.
  - Batch candidate--candidate couplings.
  - Ablation nomination, deletion, and second refit.
  - Recompute all dependent percentages and claims.
  - DONE subset: all nine Bundle-5 weak-sector values at k-star and terminal;
    the three completed strong-sector Append reference prefixes; the current
    pure-Hubbard optional-control prefixes; and the fixed-seed noise prefixes.
  - Bundle-5 corrected k-star values, ordered W--W/I--W/S--W, are:
    Append 6277/5599/5790, Plateau 8736/5599/10294, and Always
    17776/12358/14031. Their raw ledgers are retained as receipts, not printed
    as the mathematical estimator cost.
  - TODO subset: each strong-sector insertion value as its ledger is promoted,
    plus a final literal comparison of every displayed number and percentage
    against the frozen manuscript displays.

- [x] **EST-04 — Legitimate repeated Powell objectives are distinguished from
  reusable staged-selection observables.**
  - Powell evaluations at different parameter points are new Hamiltonian
    expectations and must be charged.
  - Identical parameter points are not assumed cacheable on a real QPU unless
    the result is explicitly retained within the same execution.
  - Do not apply whole-ledger value deduplication.
  - Finding: every `N_H_outer(raw)` and `N_H_refit(raw)` invocation remains
    charged. Only repeated `N_grad` and `N_metric` primitive identities at the
    same source state are suppressed. Thus the correction does not assume a
    cross-callback or cross-run QPU cache.

### Central Phase-III gain and selector-cost evidence

- [x] **SCORE-01 — Bundle 5 uses marginal Phase-III candidate gain.**
  - Evidence: the source-locked route contract records
    `joint_minus_active_only_supported_trust_v1`; every inspected candidate
    gain receipt closes
    `incremental_candidate_gain_raw = full_joint_trust_gain - active_only_trust_gain`
    and selects its nonnegative resolved part. The active-only baseline uses
    the same retained support, projected generalized solver, tolerances, and
    trust radius, and has zero additional quantum-query charge because it is a
    restriction of the already measured joint model.
  - Finding: Eq. 13's subtraction of the active-only predicted decrease is
    faithful to Bundle 5. “Marginal” here means gain attributable to admitting
    the candidate beyond the relaxation already predicted for the active
    coordinates; it is not merely the full joint quadratic decrease.

- [x] **COST-CORE-01 — Bundle 5 did not use the manuscript's stated
  after-minus-before transpilation coordinates for selection.**
  - Evidence: all twelve locally transferred Bundle-5 checkpoints—nine weak
    routes and three completed strong Append routes—record
    `phase3_backend_cost_mode = marrakesh_graph_span_v1`, an empty
    `qiskit_applied_phases` list, and selected features with
    `compile_cost_source = marrakesh_graph_span_v1`. The oracle's own receipt
    says `no_transpile = true`; all raw base/trial compiled-delta fields are
    null.
  - Executed coordinates are analytic candidate costs on a fixed logical-to-
    FakeMarrakesh embedding: twice Pauli-support size minus one for the
    two-qubit-count proxy, twice the coupling-graph span for the depth proxy,
    plus one-qubit-template, parameter, and measurement proxies. The oracle
    does not transpile the source and candidate-augmented circuits.
  - Finding: this is not an implementation-only isomorphism to
    `x(U plus A) - x(U)`. The graph-span proxy may correlate with routed
    compiled marginals, but it is a different numerical coordinate and can
    change rankings. The displayed circuit-resource tuples can still be exact
    post hoc transpilation results; that does not retroactively make the
    selection cost an exact transpilation marginal.
  - Manuscript consequence: the present methods/results language describes a
    different evaluated cost oracle. Author decision required: either rerun
    the paper-facing routes with the stated transpilation oracle, or disclose
    the graph-span selector and reserve exact transpilation for reporting.

- [x] **COST-CORE-02 — Bundle 5 used five nonzero declared cost weights, not
  the three weights stated in the manuscript.**
  - Evidence: all twelve transferred checkpoints agree on
    `(lambda_2q, lambda_d, lambda_1q, lambda_theta, lambda_shot) =
    (0.20, 0.20, 0.05, 0.05, 0.15)`. The manuscript states
    `(lambda_2q, lambda_2qd, lambda_c) = (0.30, 0.30, 0.25)` and says parameter
    and measurement coordinates carry zero weight.
  - Finding: these configurations are not algebraically equivalent. Parameter
    and shot coordinates are often population-uniform and therefore normalize
    to zero, but their nonzero weights remain in the weighted-index
    denominator; the one-qubit proxy can also vary. Neither is the one-qubit
    template coordinate identical to total compiled circuit depth.
  - Manuscript consequence: this is a protocol-level mismatch, not a wording
    preference. It joins COST-CORE-01 in the rerun-versus-disclosure decision.

- [x] **COST-CORE-03 — The relative weighting transform itself is
  mathematically faithful.**
  - Evidence: Bundle 5 centers each executed coordinate by the phase-population
    median, scales by the MAD with a numerical floor, applies
    `(2/pi) arctan`, forms the lambda-weighted index `cbar`, and multiplies the
    gain by `1 - cbar/2`. The manuscript defines
    `K = 2/(2-cbar)` and scores `gain/K`, which is exactly the same factor.
  - Equivalence allowance: the reciprocal `K` notation need not mirror the
    code variable names. The mismatch is in the coordinates and weights fed to
    the transform, not in the transform's mathematics.

- [x] **COST-CORE-04 — Earliest commutation-class representatives are a
  valid tie representative under the executed graph-span oracle, but not
  evidence of cost-minimizing transpiled positions.**
  - Evidence: insertion routes record
    `termwise_cross_component_commutation_earliest_representative_v1`. The
    graph-span candidate coordinates do not depend on the member position
    inside a commutation class, so every member ties and the earliest member is
    an argmin under that executed oracle.
  - Equivalence allowance: choosing the earliest member is mathematically
    consistent with an argmin when all class members have equal modeled cost.
  - Manuscript consequence: the stronger claim that every member was
    transpiled and the least compiled position selected is unsupported by
    Bundle 5. Exact replay for post hoc circuit reporting is a separate fact to
    audit under COST-04 and RES-01.

## Complete audit queue

### A. Core mathematical model

- [ ] **MATH-01 — Candidate-augmented energy surface.** Verify that expansion
  variables, source point, gradient, Hessian, and Fubini--Study Gram matrix
  match the executed mathematical model.
- [x] **MATH-02 — Signed change versus predicted decrease.** Eq. 11 uses
  `Delta E(z)` for the signed quadratic change, while the phase quantities use
  the same symbol for a nonnegative predicted decrease. The formulas and signs
  are correct, but the notation is overloaded. This is a readability risk, not
  evidence that the executed score has the wrong sign; reserving a lower-case
  delta for the signed model remains an optional manuscript cleanup.
- [x] **MATH-03 — Phase-0 interpretation.** `|g|` is both the ADAPT screening
  score and the maximum linearized decrease over a dimensionless unit step in
  either candidate-coordinate direction. The manuscript's “unit-step
  linearized predicted decrease” is precise and does not claim a finite
  optimized energy drop.
- [x] **MATH-04 — Marginal Phase-III gain.** Verified under SCORE-01: the
  candidate score is joint predicted decrease minus the candidate-independent
  active-only predicted decrease, and the manuscript describes the intended
  marginal quantity.
- [ ] **MATH-05 — Phase-I and Phase-II restrictions.** Verify that Phase I uses
  candidate gradient plus self metric, Phase II adds candidate curvature, and
  both are genuine restrictions of Phase III.
- [x] **MATH-06 — Nonstationary source.** All twelve transferred Bundle-5
  contracts record `measured_residual_response_v1`, and the Phase-III receipts
  carry the measured active residual into both the full and active-only solves.
  The active manuscript retains `g_theta` and states the measured-residual
  policy. No stationarity assumption is needed for the Bundle-5 score.
- [ ] **MATH-07 — Trust-region solve.** Verify retained-support projection,
  generalized optimality conditions, boundary and infeasible cases, and the
  distinction between the trust multiplier and numerical regularization.
- [ ] **MATH-08 — Trust-radius update.** Verify the exact update equation,
  expansion veto, unresolved-displacement behavior, branch locality, and
  empirical range; link closure to RHO-04.
- [ ] **MATH-09 — Accepted-refit whitening.** Verify Gram-null quotient,
  spectral support, affine chart, fixed whitening frame, warm/zero start, and
  the claim that no new Gram measurement is required.
- [ ] **MATH-10 — Numerical inverses.** Enumerate every pseudoinverse,
  eigensolve, Schur complement, ridge, damping term, and tolerance. Distinguish
  the zero-ridge retained Phase-III inverse from auxiliary batch/prune
  regularization.
- [ ] **MATH-11 — Adaptive shortlist.** Verify gamma, hard caps,
  participation-ratio rounding, zero/negative-score handling, and application
  to both HH and pure-Hubbard benchmarks.
- [ ] **MATH-12 — Stopping.** Verify the stated stopping condition against
  natural-stop and forced-horizon configurations.

### B. Candidate records, positions, and equivalence classes

- [ ] **REC-01 — Raw, quotient, and positioned populations.** Determine whether
  distinct notation is needed or current prose is mathematically unambiguous.
- [ ] **REC-02 — Cut versus position terminology.** Verify every occurrence
  against whether a circuit is being split or an insertion location selected.
- [ ] **REC-03 — Eq. 17 commutation relation.** Verify the quantified crossed
  block set, ordering conventions, and equivalence relation.
- [ ] **REC-04 — Class-invariant observables.** Verify that reused response
  primitives are invariant within each certified commutation class.
- [ ] **REC-05 — Class representative.** Verify that cost-minimizing p-star is
  selected within each class and changes classical compilation only.
- [ ] **REC-06 — Insertion policies.** Verify append, plateau-triggered, and
  always-open position domains and class recomputation timing.

### C. Resource-cost mathematics

- [x] **COST-01 — Marginal compiled cost.** Contradicted for Bundle 5 under
  COST-CORE-01: selection used a no-transpile graph-span proxy, not exact
  routed after-minus-before circuit marginals.
- [x] **COST-02 — Cost aggregation.** The robust transform and reciprocal
  factor are faithful, but the declared coordinates and weights are not;
  COST-CORE-02 and COST-CORE-03 record the distinction.
- [x] **COST-03 — Phase scoring.** Bundle 5 applies the relative cost factor in
  Phases I--III and leaves Phase 0 unweighted. Phase III uses marginal
  candidate gain. The score structure is faithful, conditional on replacing
  the manuscript's unevaluated cost-coordinate contract with the executed one
  or rerunning the stated route.
- [ ] **COST-04 — Position cost.** Verify exact replay of insertion positions
  for displayed circuits and identify any ordering approximation.
- [ ] **COST-05 — Compilation reproducibility.** Verify backend, basis,
  routing, optimization level, seed, state preparation, layout policy, and the
  role of compilation angles.
- [ ] **COST-06 — Compilation reuse.** Verify which records reuse compiled
  circuits across phases and why classical reuse does not alter estimator work.
- [ ] **COST-07 — Transpiler-free estimates.** Separate appendix estimates from
  Qiskit-compiled result resources.

### D. Optional controls

- [ ] **OPT-01 — Plateau trigger.** Verify threshold, first eligible round,
  denominator behavior, patience/hysteresis, and position-domain transition.
- [ ] **OPT-02 — Batch admission.** Verify window, subset eligibility, joint
  gradient/Hessian/Gram blocks, rank gate, joint cost, insertion ordering,
  forced-horizon singleton exception, and exhaustive versus greedy semantics.
- [ ] **OPT-03 — Generator ablation.** Verify amplitude eligibility, protection
  of the current admission, deletion loss per relieved burden, measurement
  reuse, deletion refit, acceptance gate, and plotted events.
- [ ] **OPT-04 — Beam continuation.** Verify parents, children, global cap,
  tangent distinction, Pareto filter, relative cost, selected survivor, dotted
  children, and selected-lineage estimator accounting.
- [ ] **OPT-05 — Comparator distinctions.** Verify claims distinguishing batch
  from TETRIS and ablation from Pruned-ADAPT against code and primary sources.

### E. Real-QPU representativeness

- [x] **QPU-01 — Online versus diagnostic inputs.** Bundle-5 route contracts
  and accepted-round receipts contain no ED-energy controller input or
  exact-ground-state stopping test; the routes were forced to their configured
  horizons. Same-cutoff ED values enter the external reporting projection that
  forms absolute error. Exact circuit states are used to evaluate the noiseless
  benchmark observables, not as hidden knowledge of the ground state.
- [ ] **QPU-02 — Measurement recipes.** Map each gradient, curvature, active
  Hessian block, Gram element, and cross term to measurable circuits.
- [x] **QPU-03 — Measurement reuse.** EST-01--EST-04 establish the boundary:
  a scalar can be retained across selection phases only at the same prepared
  source state with the same observable/formula identity. A changed outer-loop
  state, candidate record, or optimizer parameter point requires a new
  acquisition. No cross-run cache is assumed.
- [x] **QPU-04 — Powell objective accounting.** Each recorded Powell callback
  is charged as a Hamiltonian expectation, including repeated parameter
  points. `maxiter` is an optimizer limit, not a proxy for `nfev`; displayed
  estimator work uses observed function-evaluation receipts.
- [ ] **QPU-05 — Finite shots.** Separate exact-state results, synthetic-noise
  diagnostics, and claims that do or do not extend to finite-shot execution.
- [ ] **QPU-06 — Circuit realizability.** Verify state preparation, encoded
  generators, padding validity, symmetry, routing, and target decomposition.
- [ ] **QPU-07 — Classical overhead.** Separate compilation, commutation
  classes, eigensolves, Schur solves, batching, and branch ranking from quantum
  queries.
- [ ] **QPU-08 — Parallel branches.** Separate aggregate QPU occupancy,
  individual circuit width, and selected-lineage accounting.

### F. Operator pools and benchmark protocols

- [ ] **POOL-01 — HH pool construction.** Verify families, index ranges,
  unit-Pauli normalization, coefficient absorption, filters, deduplication,
  ordering, and sizes 948/6508.
- [ ] **POOL-02 — Pure-Hubbard intact pool.** Verify derivation from the
  decoupled HH builder, 24 ordered macros, and common use by all comparators.
- [ ] **POOL-03 — Position multiplicity.** Verify the base pool is fixed while
  the generator--position population grows with the ansatz.
- [ ] **BENCH-01 — Hamiltonians and sectors.** Verify Hamiltonians, boundaries,
  filling, spin sector, encoding, reference state, and units.
- [ ] **BENCH-02 — Phonon notation.** Verify L as lattice sites/local modes, M
  as maximum occupation cutoff, registers, and same-cutoff ED comparison.
- [ ] **BENCH-03 — Optimizers.** Verify Powell version, tolerances, maxiter,
  initialization, affine/native chart distinction, and function-evaluation
  semantics.
- [ ] **BENCH-04 — Horizons and reporting.** Verify weak 45, strong 75 axes and
  current prefixes, k-star values, and plateau-entry definition.
- [ ] **BENCH-05 — Noisy diagnostics.** Verify L = 3, U/t = 1.25, noisy versus
  exact components, finite-difference step, seeds, single-realization scope,
  and measured plateau lines.

### G. Results, figures, tables, and prose claims

- [ ] **RES-01 — Figure 2.** Trace all six trajectories, current endpoints,
  horizons, markers, and hashes to Bundle 5 only.
- [ ] **RES-02 — Table I.** Recompute circuit resources and corrected estimator
  values at k-star; classify every dash.
- [ ] **RES-03 — Figure 3.** Trace optional trajectories, reporting prefixes,
  resource matrices, child branches, deletion markers, and percentages.
- [ ] **RES-04 — Figure 4/Table II.** Trace noise errors and resources to the
  fixed-seed records and corrected estimator convention.
- [ ] **RES-05 — Abstract.** Audit every error, circuit, estimator, and noise
  statement against exact scope and comparator.
- [ ] **RES-06 — Results prose.** Check all averages, ranges, counts, horizons,
  and policy exceptions.
- [ ] **RES-07 — Discussion.** Separate interpretation from setup and verify
  causal language about selection, whitening, compactness, and noise.
- [ ] **RES-08 — Numerical precision.** Define the threshold and describe
  10⁻¹⁴–10⁻¹⁶ values as numerical precision rather than physical resolution.

### H. Pseudocode, notation, and document consistency

- [ ] **DOC-01 — Complete RA pseudocode.** Verify populations, shortlist,
  batch/beam/prune paths, refits, radius updates, and stopping.
- [ ] **DOC-02 — Notation ledger.** Audit A, p, r, equivalence classes,
  phase populations, batch coordinates, gradients, Gram/Hessian objects,
  cost factors, estimator work, M, L, k, and k-star.
- [ ] **DOC-03 — Terminology.** Standardize AVQE/ADAPT-VQE, RA-Append,
  insertion/position/cut, equivalence class, extension/control, and ansatz
  plurals.
- [ ] **DOC-04 — Internal duplication.** Locate repeated settings, equations,
  and definitions and classify each as explanatory or contradictory.
- [ ] **DOC-05 — Cross-references and ordering.** Verify equation, appendix,
  caption, algorithm, bibliography, and page-break references.

### I. Reproducibility and evidentiary boundaries

- [ ] **REP-01 — Software and compiler environment.** Inventory versions
  required to reproduce reported results.
- [ ] **REP-02 — Randomness.** Inventory transpiler, simulator, coherent-noise,
  value-noise, and optimizer seeds; separate deterministic runs.
- [ ] **REP-03 — Source maps.** Ensure every display names data provenance and
  regeneration code without reader-facing local paths.
- [ ] **REP-04 — Availability.** Record what source, processed data, raw data,
  and scripts are intended for release. Do not assume open-source availability.
- [ ] **REP-05 — Claim class.** Classify material statements as mathematical
  design, author-data result, implementation fact, or literature-backed claim.
- [ ] **REP-06 — Citation boundaries.** Verify method comparisons against
  primary sources and bound absence/priority claims.

## Immediate execution order

1. Finish RHO-04 and record the true empirical population and range.
2. Finish EST-02 through EST-04, then freeze one estimator convention.
3. Audit MATH-04 and COST-01 through COST-03 together: marginal energy gain and
   marginal compiled resource are the central scientific identity of RA.
4. Audit QPU-01 through QPU-04 so mathematical reuse is not confused with
   impossible cross-state or cross-evaluation caching on hardware.
5. Audit optional controls and displayed numerical evidence.
6. Finish notation, pseudocode, reproducibility, and claim passes.

## End-state required before manuscript decisions

- Every item is DONE or explicitly retained as an author decision.
- Every DONE item names evidence and states whether the manuscript is correct,
  insufficiently explicit, or factually contradicted.
- Numerical corrections identify source population, comparator,
  numerator/denominator, and rounding.
- Suggested manuscript actions remain consequences in this ledger and are not
  applied automatically.
- Potential reruns identify the precise unresolved scientific question. No
  rerun is launched without separate authorization.
