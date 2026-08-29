# Paper I factual, mathematical, and QPU audit

Last updated: 2026-08-22  
Audit branch: codex/paper-i-factual-audit-20260822  
Reader-facing manuscript snapshot: ADAPT---Paper-I commit b1881c4
Audit mode: evidence collection only; do not edit the manuscript from this audit.

## Route-authority correction (2026-08-22)

The earlier audit pass incorrectly treated the locally promoted Bundle-5
`b3mr` archive family (cluster 9674089) as authoritative for the Paper-I run
intended by the authors.  Those archives do contain an older compatibility
route: their checkpoints explicitly record parent shortlisting, guarded
Pauli-child exposure, fixed 8/4/1 shortlists, archival runtime splitting, and
the associated historical cost contract.  That establishes what those
archives contain, but it does **not** establish that this was the Paper-I route
actually run or intended for promotion.

The fresh bind is now complete for the earlier Paper-I HH campaign.  Cluster
9664325, package `ra_adapt_costexp_depthfix_sixregime_20260820_v2`, is an
executed global-singleton route, not the legacy hierarchy.  Its W--W Append
round-1 receipt starts Phase 0 from 948 guarded singleton Pauli words, retains
24, and records `split_parent_count=0`, `split_child_count=0`, and
`unsplit_singleton_count=24`.  Every retained record has
`split_parent=false` and `child_atom_count=0`; Phase II receives the same
singletons by `identity_on_retained_singletons_v1`.  The semantic contract
independently identifies `global_guarded_singleton_pool_v1`, the global-
singleton adapter, `compatibility_resolution_active=false`, and child-pool
expansion off.  Thus the stale execution-setting string
`shortlist_pauli_children_v1` is not evidence that this route split parents;
the executed receipt proves that it did not.

By contrast, the promoted cluster-9674089 `b3mr` W--W Append receipt records
eight split parents producing 29 Pauli children, and the inspected cluster-
9674592/B6 archive records the same 8-to-29 split.  Those are later legacy-
hierarchy reruns, not evidence that the original cluster-9664325 Paper-I
route used different search populations from AVQE.  The current reader-facing
Figure-2 provenance was switched from the authenticated Bundle-3/9664325
global-singleton evidence to the `b3mr` family, so the present source binding
must be corrected before release.

One independent axis remains: 9664325 explicitly used
`stationary_source_response_v1`, whereas 9674089 used the measured active
residual.  Therefore the existing artifacts do not establish one run that is
both the authenticated global-singleton route and the later measured-residual
route.  If the paper retains the no-stationarity method as the evaluated HH
configuration, a route-faithful measured-residual global-singleton run is
still needed.  If it reports 9664325, the results protocol must disclose that
stationary-source specialization.  This is a narrower route-version choice,
not evidence that the real 9664325 run used parent-to-child selection.

Until that bind is completed, `EST-05`, `EST-06`, `COST-CORE-01`,
`COST-CORE-02`, `POOL-01`, `POOL-03`, `POOL-04`, and any release blocker that
depends on their Bundle-5 route interpretation are historical-artifact
diagnostics rather than findings about the actual Paper-I execution.

The AVQE source facts in `EST-08` have now been reproduced independently by
Codex.  The frozen wrapper caches exact parameter-byte keys, increments its
stored `objective_calls` only on cache misses, and writes that count as
`optimizer.nfev`.  Instrumented frozen-source replays exactly reproduce the
W--W/I--W/S--W callback versus cache-miss pairs
3351/3122, 2566/2386, and 2565/2374.  The noise source likewise draws value
noise only when the oracle is invoked, so a cache hit advances neither the
oracle ordinal nor the RNG stream.  What remains provisional pending the
requested independent Claude Code review is the policy conclusion: whether
Paper I should report requested Powell callbacks or executed acquisitions,
and consequently whether the cached fixed-realization noise run is valid as
executed or must be replaced under a fresh-acquisition convention.  Do not
change manuscript estimator values or authorize noise reruns until that
convention is settled.

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
  - Manuscript status: commit `b1881c4` now reports the range over the
    “displayed trajectory prefixes,” rather than the transient archive-only
    subset available at the preceding edit, and rounds the observed maximum to
    19. That scope is supported by the source population summarized in RHO-04.
    Regenerating any displayed trajectory creates a mandatory range-recount
    dependency.

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

- [x] **RHO-04 — The trust-radius range is closed over the trajectory prefixes
  displayed in manuscript snapshot `b1881c4`.**
  - Scope: accepted/reporting lineages shown in the figures, not discarded
    beam-child states or repeated serialized copies of the same accepted
    history. This is the scope to which the manuscript's empirical range
    sentence applies.
  - Frozen Bundle-5 archive scan of the twelve checkpoints transferred on
    2026-08-22: every archive has maximum 0.25. The minima are
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
  - Finding: over the declared display scope, the minimum is
    `1.83162e-15` (W--W plateau insertion) and the maximum is `18.9614`
    (the displayed S--S plateau-insertion prefix). Thus the reader-facing
    approximation (10^{-15}\)--(19) is supported. The checkpoint payloads
    serialize repeated copies of accepted histories, but duplication does not
    affect extrema. Expansion/contraction event counts are unnecessary for
    this range claim and remain a separate diagnostic rather than an open
    manuscript dependency.

### Estimator accounting

- [x] **EST-01 — Candidate-gradient and candidate-self-metric reuse has a
  defined mathematical accounting rule.**
  - Existing evidence:
    chtc/paper_i_ra_adapt_repair_20260727/FINDINGS_ESTIMATOR_LEDGER_DUPLICATION_20260821.md.
  - Mathematical contract: a measured primitive reused by a later phase at the
    same source state is charged once, regardless of whether the current code
    recomputed it.
  - Candidate-stage correction used in the existing recount:
    N_H_outer(raw) + N_H_refit(raw) + N_grad(unique) + N_metric(unique).
    This expression repairs same-round candidate reuse only; it is not yet the
    final manuscript ledger because it retains the outer-state refreshes
    identified in EST-07.
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
  - Scope qualification: the resulting values are
    *candidate-reuse-corrected Bundle-5 artifact totals*. They are not yet
    final Paper-I \(\mathcal E\) values because EST-07 identifies an additional
    cross-round source-energy mismatch and POOL-01 makes Bundle 5 an
    authority-conflicted route.

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
    the three strong-sector Append reference prefixes presently promoted into
    the manuscript figure; the current pure-Hubbard optional-control prefixes;
    and the fixed-seed noise prefixes.
  - Interim Bundle-5 mathematical-reuse-corrected k-star values, ordered
    W--W/I--W/S--W, are:
    Append 6266/5590/5780, Plateau 8725/5590/10284, and Always
    17765/12349/14021. These values also remove the first \(k^\ast\)
    `N_H_outer` occurrences under EST-07. Their raw ledgers are retained as
    receipts, not printed as the mathematical estimator cost.
  - The independently replayed AVQE values at the same weak-sector reference
    rounds are 13779/11098/12045. Unlike the presently printed
    13538/10910/11845 values, these remove `N_H_outer` but restore every Powell
    callback suppressed by the runtime's exact-parameter cache; see EST-08.
  - TODO subset: the AVQE values outside these three weak reference prefixes,
    each strong-sector insertion value as its ledger is promoted, plus a final
    literal comparison of every displayed number and percentage against the
    frozen manuscript displays.

- [x] **EST-04 — Legitimate repeated Powell objectives are distinguished from
  reusable staged-selection observables.**
  - Powell evaluations at different parameter points are new Hamiltonian
    expectations and must be charged.
  - Identical parameter points are not assumed cacheable on a real QPU unless
    the result is explicitly retained within the same execution.
  - Do not apply whole-ledger value deduplication.
  - Reporting convention: every Powell callback is charged. Only repeated
    staged-response primitives at the same source state are suppressed.
    `N_H_outer` is not a Powell callback and is resolved separately under
    EST-07. Whether each executed route's stored `N_H_refit` telemetry contains
    every optimizer callback is audited separately under EST-08.

- [x] **EST-05 — Bundle 5's hierarchical parent-to-child estimator ledger is
  an artifact-route mismatch, not the accounting definition of current
  Paper-I RA.** Bundle 5 does not carry one unchanged singleton record
  population through the phases. It measures the fixed intact-parent
  population first and, after retaining eight parents, acquires gradients and
  geometry for the guarded singleton children emitted by those parents.
  - In W--W Append round 1, the 123-parent Phase-I population emits 29 children.
    The raw ledger closes at `N_grad=152` and `N_metric=181`, exactly exposing
    parent plus child acquisition before same-primitive reuse correction. In
    W--S Append round 1, 171 parents emit 42 children and the corresponding raw
    counts are `N_grad=213` and `N_metric=255`.
  - Authority finding: the current/intended Paper-I RA method is the staged
    flat-singleton selector printed in the manuscript. The Bundle-5 contracts
    instead authorize the legacy compatibility path
    `shortlist_pauli_children_v1`, with archival runtime splitting enabled and
    shared-flat-pool mode disabled. The executed hierarchy therefore does not
    redefine RA; it makes Bundle 5 nonrepresentative of the current method.
  - Replacement-route check: all 18 Bundle-6 warm-start protocol files retain
    the same `shortlist_pauli_children_v1` runtime split, set
    `allow_archival_phase3_runtime_split=true`, and leave
    `shared_pauli_pool_mode=off`. Although they label the candidate
    representation `single_pauli_word_v1`, Bundle 6 therefore does not remove
    the parent-shortlist/child-expansion authority conflict.
  - Consequence: do not alter the general RA estimator formula to encode this
    legacy split. Bundle-5 recounts remain valid descriptions of those
    artifacts, but their estimator savings cannot be promoted as evidence for
    the current flat-pool method without either a route-faithful rerun or an
    explicit author decision to make the hierarchy part of the paper method.

- [ ] **EST-06 — Freeze estimator values for evidence generated by the
  route-authoritative flat-singleton RA method.** If Bundle 5 is replaced,
  apply the EST-01 primitive-reuse convention to the replacement ledgers and
  regenerate every estimator value and percentage. If Bundle 5 is retained by
  author decision, derive and disclose a separate parent/child accounting
  expression rather than silently using the current singleton formula.
  - Existing-package search does not supply a substitute. A field-wise scan of
    every **tracked** JSON under the Paper-I campaign `protocols/` directories
    at source HEAD `5775f0e` found 775 protocols declaring
    `candidate_representation=single_pauli_word_v1`. Of these, 669 combine the
    legacy `shortlist_pauli_children_v1` split with the stationary-source
    response, and 28 combine that split with measured residual response. The
    26 protocols with `phase3_runtime_split_mode=off` are older pure-Hubbard
    noise protocols and still use the stationary-source response; no tracked
    protocol is an HH measured-residual, split-off flat-singleton route. Nor
    does any packaged protocol combine single-Pauli candidates, split-off
    execution, and the older explicit `shared_pauli_pool_mode=on` flag.
  - The 18 untracked Bundle-6 protocols add measured-residual cases but retain
    the same legacy child split and `shared_pauli_pool_mode=off`, as already
    established under EST-05. Including them therefore changes the measured-
    residual legacy count from 28 to 46, but does not produce a route-correct
    candidate.
  - Consequently, directory names containing “singleton” are not evidence of
    route fidelity: the older six-regime singleton packages inspected here
    still authorize parent shortlisting followed by Pauli-child expansion.
    EST-06 requires new evidence rather than promotion of one of those
    historical packages.

- [x] **EST-07 — Per-round outer-state refreshes are executed-statevector
  work, but the manuscript's mathematical ledger reuses the preceding accepted
  endpoint energy.**
  - Evidence: in the W--W Bundle-5 Append ledger, terminal raw accounting
    contains 50 `N_H_outer` occurrences. Exactly one has scope
    `energy:initial_state` and `charged=true`; the other 49 have scope
    `outer_state_refresh` and `charged=false`. At \(k^\ast=11\), the raw prefix
    contains one initial occurrence and ten uncharged refreshes.
  - Source trace: `adapt_pipeline.py` recomputes
    `energy_via_one_apply(psi_current,H)` before the gradient surface in order
    to obtain both the scalar energy and the statevector-only helper
    \(H|\psi\rangle\). On a QPU the candidate commutators are acquired directly;
    forming \(H|\psi\rangle\) is not an additional logical scalar-estimator
    primitive. The accepted Powell endpoint already supplies the next round's
    scalar source energy.
  - Comparator trace: the AVQE continuation ledger likewise records one
    `outer_energy` occurrence in every round, while the manuscript's AVQE
    formula contains only the Powell-objective and pool-gradient terms.
  - Finding: the manuscript's cross-round reuse statement is mathematically
    and QPU-faithful, but the presently promoted `S_alg` totals retain
    implementation refreshes for both methods. The current candidate-only
    correction therefore does not fully reproduce either displayed equation.
  - Convention finding: charge zero `N_H_outer` entries. The declared
    Hartree--Fock/phonon-vacuum and pure-Hubbard reference states are known
    computational-basis product states, so their source expectations are
    classically evaluable from the known Pauli Hamiltonians. Every later source
    energy is the preceding accepted Powell endpoint. This is the convention
    already expressed by the displayed AVQE and RA formulas and requires no
    cross-state or cross-run QPU cache.

- [ ] **EST-08 — Verify stored Powell work against optimizer callback counts,
  not merely against executed Hamiltonian cache misses.** The three weak AVQE
  reference prefixes are now closed; the remaining displayed AVQE prefixes
  are not.
  - RA's accepted-refit objective records every evaluation directly and stores
    SciPy's `optimizer_result.nfev`; its Bundle-5 histories close against raw
    `N_H_refit` occurrences.
  - The AVQE implementation instead memoizes exact parameter-byte keys inside
    each Powell refit. It increments the stored `objective_calls` counter and
    records `N_H_refit` only on cache misses, then writes that distinct-point
    counter to `history[].optimizer.nfev`; SciPy's original callback count is
    not serialized there.
  - The rounds 51--75 W--W AVQE continuation happens to close
    `sum(history[].optimizer.nfev)=N_H_refit=16558`, but this only proves
    agreement between two cache-miss counters. It does not prove that no
    identical-point callbacks were suppressed.
  - Frozen-source reproduction now proves that cache hits occur. The recovered
    rounds-1--50 source archive has SHA-256
    `1f949b0cc8b61dca63911832e8dc8bb32614174755ac476827956bb0812accee`;
    its Append module has SHA-256
    `f04cec3fedfb0dbcc0716f88941ada668d9832107d303dc01073313eccf531fa`,
    matching the authoritative tolerance-overlay receipt. Replaying with that
    receipt's `1e-4/1e-8` overlay exactly reproduces all three weak-sector
    energies and raw ledger prefixes at \(k^\ast=11/9/10\).
  - Ordered W--W/I--W/S--W, the raw ledgers are
    13561/10927/11864, the stored distinct-point Powell counts are
    3122/2386/2374, and SciPy actually invokes the objective
    3351/2566/2565 times. Hence the runtime cache suppresses
    229/180/191 real optimizer callbacks. The manuscript values
    13538/10910/11845 are instead the ledger's whole-prefix `S_unique`; they
    additionally suppress legitimate repeated Powell work and are not valid
    under EST-04.
  - Applying the closed convention
    \(\mathcal E=S_{\rm raw}-N_{H,{\rm outer}}+
    N_{\rm callback}-N_{H,{\rm refit}}\) gives
    13779/11098/12045. The subtraction removes 11/9/10 outer-state refreshes;
    the callback term restores only the exact-point Powell calls hidden by the
    implementation cache. The post-optimizer lookup of the returned point is
    not charged because SciPy already supplies the terminal objective value.
  - The fixed-seed noise campaign is no longer merely a probe-only source: its
    campaign manifest contains all three completed 20-round AVQE cells and
    artifact receipts. At the displayed round 10, the low/high/extreme raw
    component totals are respectively
    \((N_{H,{\rm outer}},N_{H,{\rm refit}},N_{\rm grad},N_{\rm metric})=
    (10,10606,240,0)/(10,13426,240,0)/(10,6332,240,0)\), giving the stored
    totals 10856/13676/6582. The corresponding round-20 totals are
    29249/30877/26399.
  - Those completed noise artifacts still do **not** retain SciPy's callback
    count. Their frozen source fact is commit
    `e230a588d1422af506c10a4754c81fb4f19daa8e`. At that commit,
    `append.py` increments `objective_calls` only after an exact parameter-byte
    cache miss, calls `scipy.optimize.minimize`, and then serializes
    `history[].optimizer.nfev = objective_calls` rather than
    `optimizer_result.nfev`. The ledger and history therefore agree by
    construction on distinct acquisitions, while the raw Powell callback
    count is discarded. The signed artifacts cannot recover the missing
    callback multiplicity without a controlled replay.
  - Replacement evidence should preserve both values explicitly: the raw
    SciPy callback count used by the manuscript's QPU-faithful convention and
    the distinct executed-acquisition count used to diagnose implementation
    memoization. One must not be written into the other's field.
  - This is still a live replacement-route defect at clean source HEAD
    `5775f0ebcbf985cde4698e300e6e16f52450e934`: the conventional comparator's
    `append.py` continues to serialize its cache-miss-only `objective_calls`
    as `optimizer.nfev`. By contrast, the RA accepted-refit objective records
    one `N_H_refit` occurrence on every objective invocation and the accepted
    history retains `artifacts.optimizer_result.nfev`. The missing telemetry
    is therefore comparator-specific, not an unavoidable limitation of the
    common optimizer stack. Both relevant source files are unmodified relative
    to that HEAD.
  - In the noise comparator, the cache is scientifically consequential rather
    than only an accounting shortcut. At frozen commit `e230a588`, every
    `evaluate_parameterized` call carries the next serial evaluation ordinal,
    increments `evaluation_count`, and records the resulting value-noise draw.
    An exact-parameter cache hit in `append.py` skips that oracle invocation
    and does not advance the noise state. Under the author-set EST-04
    convention that a real-QPU Powell callback is a fresh acquisition, the
    noise displays require a no-cache, callback-telemetry-complete comparator
    rerun; adding the suppressed callback count after the fact would repair
    \(\mathcal E\) but would not reconstruct the stochastic trajectory that
    fresh acquisitions would have produced.
  - Remaining closure: apply the same frozen-source callback instrumentation
    to every other displayed AVQE prefix, including the strong-HH,
    pure-Hubbard, and noisy panels. Do not infer full closure from the present
    `optimizer.nfev` field or apply whole-ledger value deduplication.

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
    routes and the three strong Append routes presently promoted into the
    manuscript figure—record
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
  - Replacement-route check: all 18 Bundle-6 protocols still set the operative
    `phase3_backend_cost_mode=marrakesh_graph_span_v1`. They also contain a
    generic `cost_backend=qiskit_transpilation_v1` field, but that declaration
    does not by itself supersede the more specific Phase-III selector mode.
    Bundle-6 result receipts must show which oracle actually supplied selected
    features; its protocol files alone do not repair this mismatch.

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

- [x] **MATH-01 — Candidate-augmented energy surface.** The manuscript and the
  Bundle-5 receipts use the same zero-insertion source chart: candidate
  amplitude plus active-coordinate displacement, evaluated at candidate
  amplitude zero and the accepted optimized source state. The archived
  derivative convention is `compiled_ansatz_exact_parameter_derivatives_v1`;
  the gradient sign follows the stated `exp(-i alpha A)` convention, the
  ordered energy Hessian retains candidate, cross, and active blocks, and the
  Gram receipt identifies the horizontal/projective Fubini--Study metric.
  The source state, logical-coordinate registry, insertion position, and
  Hamiltonian fingerprints are bound in each measured geometry receipt.
  - Finding: the reader-facing energy surface is a faithful mathematical
    representation of the executed derivative chart. The code's internal
    descent vector is the negative of the manuscript energy gradient; this is
    a sign convention, not a different model.
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
- [x] **MATH-05 — Phase-I and Phase-II restrictions.** Phase I is faithful:
  Bundle 5 uses the candidate gradient and candidate self metric in the
  one-dimensional first-order Fubini--Study trust model. Phase II does acquire
  and authenticate the signed candidate curvature `h_raw = H_alphaalpha`, but
  its executed ranking model replaces that curvature by `max(0, h_raw)` before
  calling the scalar trust solver. Thus positive curvature is included while
  negative curvature is conservatively clipped to zero.
  - Direct evidence: in Bundle-5 W--W round 1, the selected feature records
    `g_signed = 2`, `G_BB = 1`, and `H_BB = h_raw = -0.5`. The stored Phase-II
    gain is `0.5`, the first-order boundary gain at `rho = 0.25`; the signed
    quadratic restriction printed in the manuscript would instead give
    `0.515625`. The source-locked `energy_model/phases.py` has SHA-256
    `7ecf3a74...4894`, matching the audited branch, and calls
    `trust_region_drop(g, max(0,h_raw), G, rho)`; the archived helper itself
    applies the same nonnegative-curvature clipping.
  - Finding: the phases are nested in measured information and coordinate
    scope, but the active Phase-II score is not literally the signed scalar
    restriction of the printed Phase-III quadratic for negative curvature.
    This can change the Phase-II shortlist.
  - Manuscript consequence: author decision required between (a) disclosing
    Phase II as a conservative positive-curvature restriction, or (b) rerunning
    with the signed scalar model currently derived in the paper. This is not an
    implementation-detail equivalence.
- [x] **MATH-06 — Nonstationary source.** All twelve transferred Bundle-5
  contracts record `measured_residual_response_v1`, and the Phase-III receipts
  carry the measured active residual into both the full and active-only solves.
  The active manuscript retains `g_theta` and states the measured-residual
  policy. No stationarity assumption is needed for the Bundle-5 score.
- [x] **MATH-07 — Trust-region solve.** Bundle 5 uses
  `supported_metric_projected_generalized_trust_v1`. The source-locked solver
  and the audited branch file are byte-identical (SHA-256
  `655a7209...6bc04`). It diagonalizes the raw Gram only to remove unsupported
  directions, preserves the retained eigenvalues as the generalized trust
  metric, and solves the supported system without forming a Gram inverse or
  inverse square root. Empty support, eigensolve failure, failed trust-radius
  bracketing, nonfinite solutions, and materially negative predicted decrease
  are infeasible cases.
  - The code receives the negative of the physical energy gradient, so its
    internal `(H + mu G) z = g_descent` is exactly the manuscript equation
    `(H + mu G) z = -g_energy`. The predicted decrease is evaluated with the
    unshifted Hessian.
  - Numerical curvature stabilization is recorded separately as `kappa`; the
    trust-boundary multiplier is `lambda`, and the total metric shift is
    `mu = kappa + lambda`. Across all twelve transferred Bundle-5 histories,
    no selected solve had `kappa > 0` with `lambda = 0`; every stabilized
    selected solve was also boundary-active. The manuscript's one-symbol KKT
    form is therefore faithful to the executed population, while the generic
    numerical distinction remains to be documented under MATH-10.

- [x] **MATH-08 — Trust-radius update.** The mathematical update, safeguards,
  and empirical population scope are closed.
  - All twelve transferred Bundle-5 histories use
    `source_metric_inverse_sqrt_no_overlap_v1`. Every changed radius closes
    `rho_after = rho_before * sqrt(d_pred/d_real)` to serialized numerical
    precision. The largest normalized residual in the streamed audit was
    `8.1e-9`.
  - Every expansion in those archives was both boundary-active and followed a
    positive complete-refit energy decrease. No unresolved transaction changed
    the radius. Contractions apply without that expansion veto. Every
    endpoint-overlap charge is zero.
  - Trust state is cloned into each beam child and the selected child's state
    is promoted, so the update is branch-local rather than one global scalar
    shared across parallel branches.
  - Finding: the manuscript's adaptive-radius equation and its safeguard prose
    represent the executed update. Over the accepted trajectory prefixes
    presently promoted into the manuscript displays, the audited radii span
    \(1.83162\times10^{-15}\) to \(18.9614\), supporting the reader-facing
    approximation \(10^{-15}\) to \(19\). This is deliberately scoped to the
    frozen displayed prefixes rather than to jobs whose future endpoints can
    still change.

- [x] **MATH-09 — Accepted-refit whitening.** All twelve transferred
  Bundle-5 histories record `supported_fs_whitened_fixed_v1`. Each accepted
  refit quotients the raw Gram-null complement, applies the relative spectral
  support threshold, and maps the retained eigenvectors through an
  inverse-square-root metric factor into an affine Powell chart. The mapping
  explicitly permutes Phase-III response order into post-admission logical
  parameter order.
  - The inherited zero-growth state is the affine origin. A guarded Phase-III
    joint step is mapped into the whitened chart when it is a valid downhill
    initialization; otherwise the incumbent origin is used. Powell remains
    authoritative for the completed refit.
  - Every chart is fixed for one Powell invocation, discarded afterward, and
    recomputed after the next admission. The archived receipts report no
    metric-backend evaluation and zero incremental metric-query charge: the
    same-state Phase-III joint Gram is reused.
  - The executed whitening uses a `1e-9` numerical metric ridge on the retained
    support. This does not restore discarded Gram-null directions; it changes
    only the inverse-square-root scaling of retained positive modes. The
    manuscript's undamped spectral formula is the high-level map, but the
    numerical ridge belongs in the reproducibility convention audited under
    MATH-10.

- [x] **MATH-10 — Numerical inverses and regularizers.** **The evaluated
  numerical conventions are now separated by purpose; the manuscript's
  blanket “inverted without a ridge” statement is false for accepted refits,
  while a blanket claim that every inverse is damped would also be false.**
  - Phase I: the first-order Fubini--Study scalar score returns zero when its
    measured candidate self metric is nonpositive, so it never divides by an
    unresolved \(G_{\alpha\alpha}\). Phase II uses a `1e-12` metric floor in
    its scalar trust bound and clips negative candidate curvature to zero as
    recorded in MATH-05. These are evaluated score rules, not properties of
    the signed scalar model currently printed in Appendix A.
  - Singleton Phase III and its candidate-independent active-only baseline use
    the **same** raw-Gram supported projected-generalized solver, relative
    support tolerance `1e-6`, and current trust radius. They apply no Gram
    ridge and construct no Gram inverse, pseudoinverse, or inverse square
    root. A `1e-9` energy-resolution floor, enlarged when machine error is
    larger, bounds curvature stabilization, negative-gain comparison, and
    trust bracketing. The actual KKT metric shift remains data-dependent; it
    is not a fixed `1e-9` ridge added to \(\mathbf G\).
  - The displayed combinatorial batch route uses the same supported
    projected-generalized solve on the full active--batch blocks with support
    tolerance `1e-6`. Its receipts serialize
    `metric_regularization=1e-9` only as a configured **inactive** legacy
    parameter and explicitly use zero metric regularization in the retained
    support. Its `energy_regularization=1e-9` has the same resolution/curvature
    role as above. Thus the Discussion sentence that the batch linear solve
    “uses a ridge of \(10^{-9}\)” misidentifies an energy tolerance as a Gram
    ridge.
  - Accepted refit is different by design: it applies the same relative
    support threshold `1e-6` and adds `1e-9` to the retained positive Gram
    eigenvalues before inverse-square-root whitening. The forward affine map
    uses that spectral factor. A `pinv(...,rcond=1e-12)` appears only in the
    inverse coordinate map used to express an optional warm start; it is not
    the physical forward optimizer chart or the Phase-III response solve.
  - The plotted ablation nomination first projects its measured Hessian and
    Gram to symmetric positive-semidefinite matrices, forms
    \(\mathbf H+10^{-6}\mathbf G\), and adds a separate
    \(10^{-6}I\) ridge to the surviving-coordinate Schur block before a
    NumPy pseudoinverse. The route uses the stationary-survivor right-hand
    side described in OPT-03. These values regularize the classical
    nomination surrogate only; the measured delete--refit energy remains the
    acceptance authority.
  - The all-models-infeasible Gram-novelty fallback reuses an already measured
    reduced Gram block and solves it with a base ridge `1e-6`, multiplying the
    ridge by ten on failure for at most twelve attempts. It records the actual
    `ridge_used` in each fired receipt, charges zero quantum queries, and
    canonicalizes only the fallback ranking score at the configured
    `cheap_score_eps=1e-12` resolution. Ordinary Phase-III energy scores are
    not quantized this way.
  - The population-relative cost transform floors a vanishing MAD scale at
    `1e-12`. The trust-radius update uses source-Gram quadratic forms directly
    and no inverse. The fixed-noise gradient uses a central energy difference
    with step `0.1`; it introduces no additional matrix inverse or damping
    parameter, while RA's Gram geometry remains exact in that diagnostic.
  - Manuscript consequence: reproducibility prose can state these conventions
    compactly by role—support cutoff, Phase-III energy resolution, accepted-
    refit whitening ridge, ablation ridge, fallback ridge, and cost scale
    floor. It should not compress them into one claimed global damping range,
    because they regularize different mathematical objects and some principal
    solves are deliberately ridge-free.
- [x] **MATH-11 — Adaptive shortlist.** **Material evaluated-protocol
  mismatch.** The adjacent-score-ratio frontier, inverse-Simpson effective
  population, nearest-integer rounding, positive-score restriction, tie-shell
  handling, and hard cap printed in the manuscript agree with the current
  `adaptive_phase_shortlist.py` implementation. They do not describe the
  reported Paper-I runs.
  - All 12 transferred Bundle-5 HH checkpoints record
    `adaptive_shortlist=null` and an empty adaptive score population in every
    Phase-I--III receipt. The effective retained widths are fixed at `8/4/1`
    in every round (one S--W always-insertion Phase-II round retains three
    because only three records remain). Append Phase-I populations are 123 for
    \(M=3\) and 171 for \(M=7\); insertion policies expand the positioned
    population but retain the same fixed widths.
  - All four \(L=3\) optional-control v10 records likewise have no adaptive
    shortlist receipt. Their usual fixed widths are `24/12/1`; batch expands
    the Phase-III subset population and beam can reduce the available Phase-I
    population, but neither activates the printed adaptive rule.
  - The manuscript statement that both benchmark families use the same
    adaptive shortlisting with `gamma=0.9` is therefore false for the displayed
    evidence. This is not cured by mathematical equivalence: a score-dependent
    width can change which records reach later phases. Author decision:
    disclose the fixed evaluated widths while retaining adaptive shortlisting
    as a method capability, or rerun with the adaptive policy.
- [x] **MATH-12 — Stopping.** **Method/evaluated-route scope mismatch.** The
  generic natural-stop statement is a valid algorithmic option, but it does
  not describe the Bundle-5 reporting trajectories.
  - The staged HH configuration disables the ordinary `eps_grad` and
    `eps_energy` termination gates and is driven to a per-regime source-locked
    horizon: 50 accepted rounds for the weak-cutoff routes and 75 for the
    transferred strong-cutoff Append routes. The checkpoints therefore end
    with `success=false` and no natural `stop_reason`; this is configured
    horizon completion, not algorithmic convergence.
  - Every Bundle-5 route authorizes the zero-query deferred-Gram fallback
    `collective_span_novelty_over_cost_v1`. When every supported-coordinate
    energy model is infeasible, it may admit one Gram-novel direction even
    though the clipped marginal Phase-III candidate gain is zero. Across the
    12 transferred histories the fallback fires in 126 accepted rounds. For
    example, W--W Append fires in rounds 27, 29, 32, 35--44, 46, 48, and 50;
    the selected records have zero `phase3_incremental_candidate_gain` but a
    positive geometry-expansion score, and the subsequent complete refit can
    still lower the measured energy.
  - Thus “no feasible Phase-III record carries a positive predicted decrease”
    is not the operational stopping rule for these evaluated routes. The
    manuscript must distinguish the generic natural stop from forced-horizon
    evaluation with the all-models-infeasible Gram-novelty fallback. This can
    be disclosed without implementation-level detail: the fallback reuses the
    already measured Gram geometry and incurs no extra quantum queries.

### B. Candidate records, positions, and equivalence classes

- [x] **REC-01 — Raw, quotient, and positioned populations.** The current
  notation is mathematically sufficient without a separate raw-population
  symbol: `P(A)` is explicitly the set of commutation-inequivalent positions,
  so `R_k` is already the quotient-representative population. The later map
  from a class representative to `p-star` changes only its positioned record.
  A three-object notation would expose implementation stages more literally
  but is not required for correctness.
- [x] **REC-02 — Cut versus position terminology.** Active text uses
  “position” for the selectable integer and “factors/factoring” when splitting
  the circuit into prefix and suffix. This is the clean distinction; no
  reader-facing `cut` object is required.
- [x] **REC-03 — Eq. 17 commutation relation.** Verified. Positions 0 through
  n lie before, between, and after the n active blocks; moving from j to j-prime
  crosses exactly blocks `min(j,j-prime)+1` through `max(j,j-prime)`. The code
  uses the same componentwise all-pairs Pauli-commutation certificate and
  partitions contiguous positions at each noncommuting crossing.
- [x] **REC-04 — Class-invariant observables.** The componentwise certificate
  is sufficient for the implemented ordered Pauli-rotation blocks to commute
  exactly. Crossing only certified blocks leaves the candidate-augmented
  unitary, source state, and response derivatives unchanged. Reusing one set
  of response primitives per class is therefore mathematically faithful; no
  empirical approximate-commutation reuse was found in this route.
- [x] **REC-05 — Class representative.** COST-CORE-04 records the evaluated
  result: Bundle 5 takes the earliest representative, which is a valid cost tie
  under its position-independent graph-span oracle. It did not transpile all
  class members to establish the stronger manuscript claim.
- [x] **REC-06 — Insertion policies.** Route contracts and accepted histories
  verify append-only endpoint restriction, plateau-gated opening, always-open
  commutation-class exposure, and recomputation from the current accepted
  ansatz each round. One wording defect remains: “Always-insertion admits from
  every class” should mean it *permits candidates* from every class; only the
  selected record is admitted in a singleton round.

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
- [x] **COST-04 — Position replay for reported circuits.** The presently
  tabulated Bundle-5 weak-sector resource tuples reconstruct the exact accepted
  circuit order, rather than truncating the terminal order.
  - For every one of the nine 50-round weak Bundle-5 logs, each recorded
    insertion position lies in \(0,\ldots,k-1\). The three RA-Append logs use
    position \(k-1\) in every round. For Plateau and Always, the reporting
    generator starts from the empty list and applies `seq.insert(p,word)` at
    each recorded round before compiling the requested prefix. No invalid
    position is encountered, so its defensive append fallback never fires.
  - Table I presently reports no strong-sector insertion-policy resource
    tuple; the strong RA-Append tuples use acceptance order, which equals the
    recorded append order. Figure 2 carries energy trajectories only.
  - Finding: there is no final-ordering approximation in the currently
    displayed HH circuit tuples. Any newly promoted insertion resource tuple
    must repeat this position-validity and prefix-replay check.
- [x] **COST-05 — Compilation convention.** The reporting compiler constructs
  the full accepted-prefix circuit, includes computational-basis reference
  preparation, and assigns every structural Pauli rotation the common nonzero
  angle \(1.0\). It then calls Qiskit's transpiler on the local exact
  156-qubit `FakeMarrakesh` backend at optimization level 1 with
  `seed_transpiler=7`; routing and SWAP decomposition are therefore included.
  The resolved backend exposes `id/sx/rz/x/cz` as its unitary basis operations,
  consistent with the manuscript description.
  - No `initial_layout`, `layout_method`, or `routing_method` is supplied, so
    Qiskit chooses them under its defaults and the fixed seed. The tuple
    receipts record target name, exact-fake resolution, optimization level,
    seed, routed status, and compiled width, but not a serialized backend
    target, transpiled circuit, chosen layout, or package inventory. Exact
    byte-for-byte regeneration therefore still depends on REP-01 freezing the
    Qiskit/runtime versions and backend snapshot.
  - These are structural resource counts at a fixed placeholder angle, not
    recompilations at each optimized variational amplitude. That convention
    prevents zero-angle removal and keeps comparisons amplitude-independent,
    but it must not be described as the circuit returned at its numerical
    optimum.
- [x] **COST-06 — Classical cost reuse.** The manuscript's cache principle is
  mathematically sound: once a generator--position circuit has been compiled,
  later phases may reuse its classical resource tuple without changing any
  estimator count. The current exact-transpilation route has a hashed
  candidate-position cache identity for this purpose. **Bundle 5 did not
  exercise that route:** it evaluated deterministic graph-span features and
  records no Qiskit-applied selection phase. Its repeated phase costs are
  therefore reused analytic features, and the sentence that fresh
  transpilation was incurred at Phase I is false for the reported HH
  artifacts.
- [x] **COST-07 — Transpiler-free versus reported costs.** The appendix's
  support-count/graph-span formulas are analytic cost estimates; Table I's
  \(N_{2q}/D_{2q}/D_c\) values are separate post hoc Qiskit compilations of the
  accepted prefixes. They must not be presented as the same measured
  coordinate. Moreover, COST-CORE-01/02 show that Bundle 5 actually selected
  with the graph-span family and five nonzero weights, despite the manuscript
  describing the analytic coordinates as optional and zero-weight. Thus the
  reporting tuples are valid circuit diagnostics for those accepted ansätze,
  but they do not prove that the selector optimized the displayed compiled
  resources.

### D. Optional controls

- [x] **OPT-01 — Plateau trigger.** The equation and evaluated policy agree.
  At the start of a controller round, the route takes the decrease from the
  immediately preceding accepted full refit and divides it by the mean of all
  still-earlier accepted-refit decreases. It opens the full
  commutation-reduced position domain exactly when that ratio is strictly
  below \(\tau=10^{-4}\); otherwise it permits the append endpoint only.
  - A ratio is unavailable through the first two accepted transitions and
    whenever the complete prior history is unavailable or its cumulative
    decrease is nonpositive. In those cases the domain remains closed, so
    there is no zero-denominator division. The earliest eligible controller
    round is therefore round 3.
  - Bundle 5 records patience one and no hysteresis or latch. The predicate is
    recomputed each round and may close again after opening. This occurs, for
    example, in W--W: the domain first opens at round 11, closes at rounds
    12--13, opens at 14--22, closes at 23--25, and then opens thereafter.
    The manuscript's phrase “for that round” is therefore precise; prose
    implying permanent activation would be false.
  - The three weak plateau histories presently promoted into the manuscript
    share this policy. Their first
    open rounds are 11, 12, and 5 in W--W, I--W, and S--W. The main noiseless
    policy consumes only accepted variational energies, not the ED reference.
- [x] **OPT-02 — Batch admission.** The affine-chart \(L=3\) batch artifact and
  its source-locked execution commit `8501ffd` establish the evaluated rule.
  It takes the first six records of the Phase-II score order, exhaustively
  enumerates generator-distinct subsets of sizes one through three, and keeps
  only subsets whose supported joint solve exists and whose joint Gram rank
  exceeds the active rank by the subset cardinality. With six distinct
  records this is at most \(6+15+20=41\) subset solves per round. The joint
  workspace contains the active and batch gradients, the full active--batch
  Hessian and Gram blocks, and the batch candidate--candidate off-diagonal
  entries; every required candidate pair is measured once for the search and
  the winning pairs are an attribution subset, not a second acquisition.
  - The accepted object is an **ordered** tuple, even though subset
    enumeration is combinatorial: the immutable Phase-II order fixes the
    member order, and the transition commits that fixed order. At the append
    endpoint the reported effective positions consequently increase as each
    preceding member is inserted. Writing \(B\) as an unordered set is
    harmless only if this deterministic order is stated in prose; it is not
    generally irrelevant for noncommuting generators.
  - The evaluated cost denominator is not the undefined manuscript phrase
    “\(K_3(B)\) combines the member records' resource contributions.” For the
    canonical ranked-Phase-II route, the source computes
    \(1+\sum_{r\in B}\kappa_2(r)\), where each stored
    \(\kappa_2\) is the nonnegative Phase-II cost excess. The manuscript
    should either define its batch factor to be exactly this quantity or use
    a distinct symbol; it must not imply that the singleton Phase-III
    median-relative factors are recomputed over batches.
  - The plotted batch route accepted cardinalities \(3,3,2\), followed by
    eight singletons, and ended after accepted round 11 because round 12 had
    no rank-feasible subset. The executed artifact contains no forced-horizon
    rank-null singleton exception. A current-code exception or a later forced
    run must not be retroactively attributed to this figure.
  - The implementation also contains the advertised score-ordered greedy
    alternative, but the plotted route used the exhaustive combinatorial
    policy. This distinction is manuscript-level and should remain explicit.
- [x] **OPT-03 — Generator ablation.** The affine-chart \(L=3\) ablation
  artifact and frozen source commit `8501ffd` establish the evaluated rule.
  It protects the current admission (`protect_steps=1`) and applies a strict
  amplitude screen
  \(\lvert\theta_p\rvert<10^{-3}\) to older coordinates. It then ranks one
  measured delete--refit trial by metric-regularized Schur loss per compiled
  burden relieved, deletes the nominee, performs a second complete
  Fubini--Study-chart Powell refit, and commits the deletion only from the
  measured refit energy.
  - The nomination is **not measurement-free** and does not merely reuse the
    admission response model. At the post-admission endpoint the executed
    route records a Hamiltonian anchor, active gradients, and the full active
    Gram/Hessian scaffold for nomination. The surviving ansatz then incurs
    its own gradient, Gram, Hamiltonian-anchor, and Powell-objective work for
    the delete--refit trial. The manuscript sentence claiming “no additional
    response measurements” is false for the plotted artifact.
  - The plotted route uses
    \(\mathcal Q=\mathbf H+10^{-6}\mathbf G\), an additional
    \(10^{-6}\) linear-solve ridge, and the
    `stationary_gw_zero_v1` Schur solve. The supplied gradient is therefore
    recorded but is not inserted into the Schur right-hand side. The printed
    \(g(p)\beta_p\) term describes a gradient-corrected deletion model, not
    the model that nominated the plotted deletions. This is a scientific
    formula mismatch, not an omittable implementation detail.
  - The measured acceptance rule permits at most \(10^{-8}\) energy
    regression. Its one-half retained-round-gain guard is active only when
    the just-admitted gain is at least that scale; it does not constrain
    numerical-floor admissions below the activation threshold. The surrogate
    never certifies acceptance.
  - Rounds 10--15 each accepted one deletion; the source labels repeat, so
    these are six accepted events, not six distinct generators. At the
    plotted round-12 prefix, three deletion events have occurred. Their trial
    estimator work is included in the displayed corrected lineage count.
- [x] **OPT-04 — Beam continuation.** The affine-chart \(L=3\) beam artifact
  and source-locked commit `8501ffd` agree with the manuscript's principal
  beam mathematics. The route retains at most four parents, proposes at most
  three state-dependent tangent-distinct siblings per parent, and therefore
  executes at most twelve children in a round. Tangent distinction is tested
  against the current parent's real Fubini--Study tangent span; it is not a
  label-only uniqueness test.
  - Each child is completely refitted before comparison. Its complete
    accepted prefix is compiled under the route's FakeMarrakesh contract and
    priced with the declared \(N_{2q},D_{2q},D_c\) weights. Strictly
    energy-and-cost dominated children are removed. The remaining costs are
    median-centered and MAD-scaled (with a mean-absolute-deviation fallback
    when MAD vanishes), mapped to the same bounded relative factor used by
    the paper, and ranked by cumulative descent from the common initial
    energy divided by that factor. Up to four leading children seed the next
    round; the leading ranked child is the displayed lineage.
  - The receipts exhibit the intended energy--cost tradeoff rather than a
    hidden energy-only choice. At round 15, for example, the selected child
    has energy \(-2.141493564271407\) and compiled burden \(141.35\), while
    several children have marginally lower numerical-floor energies but
    burdens near \(255\)--\(260\); the relative score chooses the cheaper
    undominated prefix. This is direct evidence that the Pareto/resource
    objective affects branch selection.
  - The plotted route executed 111 children. Its terminal selected-lineage
    estimator work is 9,822, whereas aggregate work over all executed
    children is 80,241 and discarded-child work is 70,419. The figure's solid
    resources and quoted estimator count are lineage-local; the dotted green
    traces are parallel child refits. Any claim about total QPU demand must
    use the aggregate quantity instead.
- [x] **OPT-05 — Comparator distinctions.** The claimed batch distinction is
  supported by the primary TETRIS-ADAPT-VQE paper
  (`arXiv:2209.10562`, Phys. Rev. Research 6, 013254): TETRIS adds the
  largest-gradient operator and then successively lower-gradient operators
  only when each has qubit support disjoint from every operator already added
  in that layer. The evaluated RA batch route instead permits overlapping
  supports and tests generator identity, supported joint solvability, and
  incremental Gram rank. “No support-disjointness condition” is therefore a
  precise distinction; it must not be inflated into a claim that RA has no
  feasibility gate.
  - The primary Pruned-ADAPT-VQE paper (`arXiv:2504.04652`, J. Chem. Theory
    Comput. 21, 8120 (2025)) uses an amplitude-and-relative-position decision
    factor, compares the winning amplitude with a dynamic threshold derived
    from recent amplitudes, deletes without reoptimizing the surviving
    parameters, and then evaluates the energy. Its advertised negligible
    pruning overhead follows from that no-refit design. RA ablation is
    materially different: it uses a metric-Schur/cost nomination followed by
    a measured complete delete--refit trial and an energy-acceptance gate.
    This comparison is defensible, but RA must not inherit Pruned-ADAPT's
    “cost-free” characterization: OPT-03 proves that the plotted RA route adds
    response measurements and optimizer work.

### E. Real-QPU representativeness

- [x] **QPU-01 — Online versus diagnostic inputs.** Bundle-5 route contracts
  and accepted-round receipts contain no ED-energy controller input or
  exact-ground-state stopping test; the routes were forced to their configured
  horizons. Same-cutoff ED values enter the external reporting projection that
  forms absolute error. Exact circuit states are used to evaluate the noiseless
  benchmark observables, not as hidden knowledge of the ground state.
- [x] **QPU-02 — Measurement recipes.** **Formula-level measurability is
  verified; the reported unit is a logical scalar, not an executable physical
  circuit/shot count.**
  - For a Hermitian dressed generator \(\widetilde A_u\), the gradient is the
    expectation of the Hermitian commutator observable
    \(i[\widetilde A_u,H]\). After Pauli decomposition it is measurable by
    ordinary Pauli-basis circuits; no statevector \(H|\psi\rangle\) object is
    required on hardware.
  - The printed Gram entry is the real covariance
    \(\tfrac12\langle\{\widetilde A_u,\widetilde A_v\}\rangle-
    \langle\widetilde A_u\rangle\langle\widetilde A_v\rangle\). Its Pauli
    products can likewise be measured directly, or the same overlap can be
    obtained with derivative-overlap circuits. Symmetry permits one acquisition
    per unordered coordinate pair at the logical-scalar level.
  - The energy Hessian is a second derivative of the prepared-circuit energy.
    For unit-Pauli coordinates it admits exact parameter-shift energy circuits;
    more general intact generators require a generalized shift rule, an
    ancilla overlap construction, or a declared finite-difference protocol.
    The candidate self term, candidate--active cross terms, and active block
    are instances of this same construction.
  - Implementation evidence: the noiseless benchmark backend forms analytic
    statevector tangents and evaluates overlaps with \(H|\psi\rangle\); the
    estimator ledger records abstract primitive identities such as
    `coordinate_energy_gradient_v2` and `energy_hessian_v2`. It does not emit
    the Pauli groupings, shifted circuits, coefficients, or shot allocation
    that would realize each scalar on a QPU.
  - Finding: the manuscript's claim that these quantities are
    circuit-measurable is mathematically supportable, but \(\mathcal E\) is
    only a logical-scalar ledger. Its present sentence that each call is a
    “new QPU acquisition” is defensible only if *acquisition* means one
    requested scalar estimator, not one circuit execution or one shot. Before
    claiming executable hardware-query counts, freeze one recipe per primitive
    family—especially the intact-generator Hessian—and state how many physical
    circuits, Pauli groups, and shots realize one logical invocation. This is
    a reader-facing scope correction, not an unresolved audit calculation.
- [x] **QPU-03 — Measurement reuse.** EST-01--EST-07 establish the boundary:
  a scalar can be retained across selection phases only at the same prepared
  source state with the same observable/formula identity. A changed outer-loop
  state, candidate record, or optimizer parameter point requires a new
  acquisition. No cross-run cache is assumed.
- [ ] **QPU-04 — Powell objective accounting.** The intended convention charges
  each optimizer callback as a Hamiltonian expectation, including repeated
  parameter points; `maxiter` is only an optimizer limit. EST-08 proves that
  the AVQE runtime's stored `optimizer.nfev` is a memoized distinct-point
  counter rather than SciPy's full callback count. Frozen-source replay closes
  the three weak reference prefixes and supplies their corrected values. The
  completed noise artifacts additionally prove that the two counts were not
  serialized separately even in the later pure-Hubbard comparator route. The
  remaining AVQE displays must therefore undergo the same recount—or be
  replaced by telemetry-complete evidence—before their estimator work can be
  treated as QPU-faithful callback work. For noisy evidence, replacement is
  required rather than a bookkeeping-only recount because cache hits also
  suppress noise-oracle evaluations and RNG advancement.
- [x] **QPU-05 — Exact-state, synthetic-noise, and finite-shot scope.**
  - The main HH and noiseless \(L=3\) trajectories use exact statevectors and
    contain no finite-shot or device noise. Their \(\mathcal E\) is a logical
    invocation count, not a shot count.
  - The six noise cells are one fixed realization per method/noise level on the
    six-qubit \(L=3\), \(U/t=1.25\) benchmark. Candidate gradients and Powell
    objectives are noisy; RA's geometry/Gram data remain exact; plotted errors
    are exact same-cutoff diagnostics evaluated at the noisy-returned
    parameters. Gradients use central noisy-energy differences at step 0.1.
  - The diagnostic combines gate-local depolarizing channels, frozen
    coherent Pauli overrotations, and independent scalar Gaussian value noise.
    The recorded value/coherent/simulator/transpiler seeds are
    702688422/20260609/7/7. The effective scalar-noise parameter is not a
    physical shot count, and no multi-seed dispersion or confidence interval
    is available.
  - Finding: the current reader-facing labels “fixed-seed synthetic-noise
    diagnostics” and the abstract's level-specific observations respect this
    scope. They support no universal hardware-robustness or finite-shot claim.
    The actual terminal ordering is RA lower at low and extreme perturbations,
    AVQE lower at high perturbation.
- [x] **QPU-06 — Accepted-ansatz circuit realizability.** **Verified separately
  from derivative-measurement realizability in QPU-02.**
  - The HH reference is a computational-basis Hartree--Fock determinant
    tensored with boson vacuum. The reported cutoffs \(M=3\) and \(M=7\)
    provide exactly four and eight local occupation states, so their two- and
    three-qubit binary registers contain no unused padding states.
  - The manuscript method's HH executable directions are canonical unit Pauli
    words. Their exponentials have standard basis-change/CNOT-ladder/Rz
    decompositions, and the pool builder applies fixed fermion-sector and
    legal-boson-space guards before admission.
  - The \(L=3\) optional-control checkpoints serialize every admitted intact
    block as an ordered `termwise_product` of physical Pauli rotations. The
    resource generator rebuilds that exact ordered product, includes the
    reference-state preparation, transpiles it to FakeMarrakesh at optimization
    level 1 with seed 7, and checks the resulting \(N_{2q}/D_{2q}/D_c\) against
    the stored compile receipt. Thus “intact” describes selection and shared
    parameterization, not an undecomposed matrix exponential sent to hardware.
  - Finding: the accepted ansatz circuits themselves are realizable and their
    displayed post hoc resources have an executable target decomposition. This
    does not validate Bundle 5's selector-cost oracle (COST-CORE-01), nor does
    it supply the response-observable circuits still required by QPU-02.
- [x] **QPU-07 — Classical overhead.** The manuscript's logical-estimator
  boundary is mathematically coherent: Pauli commutation classes, robust cost
  normalization, transpilation or graph-span evaluation, retained-support
  eigensolves, pseudoinverses, trust-region/Schur solves, subset enumeration,
  Pareto filtering, and branch ranking are classical. Quantum work enters only
  when acquiring response scalars or Hamiltonian objectives. This accounting
  intentionally omits potentially material wall-clock costs such as one
  compilation per candidate position and combinatorial/beam orchestration; it
  must be described as quantum-query work, not total computational cost.
- [x] **QPU-08 — Parallel branches.** **Selected-lineage accounting is
  verified; physical concurrency is not measured.**
  - The beam application contract retains up to four parents and generates
    three children per parent, capped at twelve children per round. The
    accepted-run evidence records exactly 12 executed children in rounds 10,
    11, 14, and 15, and fewer in the remaining rounds.
  - Figure 3's solid beam trajectory and resource matrix follow the selected
    branch. Its accumulated \(\mathcal E\) is reconstructed from the active
    prefix's branch-local `executed_query_delta`; rejected and surviving
    sibling refits are omitted by design. The dotted trajectories visualize
    those additional executed children without charging them to the selected
    lineage.
  - Each child remains a six-qubit circuit, so branch count never enlarges
    circuit width. However, the local campaign does not establish that twelve
    children were executed concurrently on twelve QPUs or partitions. The
    statement that they *may* be parallelized is an algorithmic scheduling
    observation; aggregate executed QPU work and realized concurrency are not
    reported resources and must not be inferred from branch-local
    \(\mathcal E\).

### F. Operator pools and benchmark protocols

- [x] **POOL-01 — Bundle-5 pool authority conflict.** **Material
  evaluated-protocol mismatch identified; the legacy executed route is not
  adopted as the definition of current Paper-I RA.**
  - The fixture inventories verify three distinct objects at each cutoff:
    123/171 ordered intact parent templates for \(M=3\) and \(M=7\),
    102/148 parents
    executable directly under the macro contract, and a globally projected,
    symmetry-guarded, deduplicated union of 948/6508 canonical singleton Pauli
    directions. Thus the printed 948/6508 counts are genuine, but they are not
    the population RA scores from start to finish.
  - Every inspected Bundle-5 receipt records `pool_type=full_meta`, pool size
    123 or 171, `phase3_runtime_split_mode=shortlist_pauli_children_v1`, and
    `shared_pauli_pool_mode=off`. Its explicit phase order is parent Phase-I
    shortlist, runtime split, child Phase II, then child Phase III. Round 1
    retains eight parents and emits 29 children at \(M=3\), or 42 children at
    \(M=7\); four children survive Phase II and one is admitted from Phase III.
  - All 18 Bundle-6 protocol files preserve that same runtime split and keep
    shared-flat-pool mode disabled. Bundle 6's warm-start change is therefore
    not, by itself, a route-faithful replacement for the flat-singleton method.
  - Runtime children are normalized modulo global scalar/sign, filtered by the
    fixed spin-resolved sector and legal boson code space, and globally
    deduplicated across retained parents. The selected generator receipts show
    unit-norm Pauli directions and preserve all contributing parent identities.
  - AVQE's accepted-word records are singleton Pauli strings. Its first-round
    estimator totals are consistent with screening all 948 or 6508 singleton
    gradients before Powell. This establishes the population mismatch in the
    executed comparison; it does not establish that hierarchical exposure is
    part of current RA.
  - Authority finding: the current Paper-I method defines one fixed executable
    singleton-Pauli pool screened through Phases 0--III without a representation
    change. Bundle 5 explicitly ran the older parent-shortlist/child-expansion
    compatibility path. The correct audit classification is therefore
    manuscript--artifact authority conflict, not “the manuscript should
    redefine RA as hierarchical.”
  - Manuscript consequence: the printed method and the promoted Bundle-5
    evidence cannot both remain authoritative without an author decision.
    Either rerun the comparison with the current flat singleton route or
    deliberately reinstate and describe the hierarchy. Until then, Bundle-5
    estimator savings and performance claims are ineligible as evidence for
    the current RA implementation.
- [x] **POOL-02 — Pure-Hubbard intact pool.** The mathematical electronic
  restriction and the executed 24-generator inventory agree.
  - The locked \(L=3\) inventory contains, in order, eight UCCSD singles and
    doubles, four spin-resolved nearest-neighbor hopping macros, three onsite
    interaction macros, and nine diagonal Hamiltonian Pauli terms. Its ordered
    label and polynomial hashes are `80f30f02...ed574` and
    `7e2f8489...03e03c`. The sector guard removes the full Hamiltonian macro
    and its eight off-diagonal Hamiltonian Pauli children, leaving exactly the
    24 labels described in the appendix.
  - The implementation does not literally instantiate an HH pool and then
    delete phonon tensor factors. It invokes the same generic `full_meta`
    construction machinery at the pure-Hubbard problem request. The
    manuscript's displayed electronic restriction is therefore a faithful
    mathematical derivation of the retained operator groups, not a claim about
    the builder's procedural call sequence.
  - The four noiseless RA control cells bind the same 24-count, ordered-label,
    and ordered-polynomial hashes. The noise comparator is constructed with
    the identical \(L=3\) macro adapter for both AVQE and RA. Thus pool
    expressibility and ordering are controlled within the pure-Hubbard
    comparisons, even though RA and AVQE use different selection/refit
    pipelines.
- [x] **POOL-03 — Position multiplicity.** The parent inventory and global
  singleton-child universe are constructed deterministically from fixed
  benchmark settings, while the exposed generator--position population changes
  with the current ansatz and insertion policy. Append keeps one position per
  exposed identity; plateau and always-insertion add commutation-inequivalent
  positions. In Bundle 5 the *child population actually measured in later
  phases* also changes with the retained-parent shortlist. Therefore “fixed
  pool” is accurate only for the underlying inventories, not for the
  per-phase record population.
- [x] **POOL-04 — Route-authoritative flat singleton pool.** **The current
  method-capability contract is exact and shared by RA and AVQE; this closure
  does not repair the Bundle-5 evidence mismatch in POOL-01.**
  - At clean builder source `5775f0e`, RA's
    `GlobalSingletonGradientPhase0CandidateAdapter.executable_pool` and the
    conventional singleton AVQE facade's `_append_inventories` both supply the
    result of `build_guarded_single_pauli_pool`. RA exposes the whole result in
    Phase I and applies identity exposure to its retained singletons in Phase
    II; AVQE scans the same whole result directly. Neither path performs a
    parent-to-child representation change after selection begins.
  - A fresh, serial, cache-disabled reconstruction compared the complete
    `CandidateRecord.manifest_row()` sequences actually returned to these two
    algorithm facades for all six benchmark requests. Counts, ordered labels,
    ordered pool receipts, and full manifest-row hashes are identical in every
    RA/AVQE pair:

    | regime | count | ordered-pool SHA-256 | manifest-row SHA-256 |
    |---|---:|---|---|
    | W--W | 948 | `74880d21...fce84` | `28bfe85d...854c` |
    | I--W | 948 | `816dfa97...e67f` | `d85dab71...ae5` |
    | S--W | 948 | `62a24f68...e90c` | `4e186444...9301` |
    | W--S | 6508 | `2b7416a8...52c0` | `0eef9a33...786` |
    | I--S | 6508 | `078aa896...ed1a` | `3f5e7945...824` |
    | S--S | 6508 | `7a0e3dac...8c59` | `aa2d3858...ee66` |

    The shared ordered-label hashes are `02995a2c...6906` at \(M=3\)
    and `07947805...cb7` at \(M=7\). The full ordered labels in the resolved
    protocols therefore freeze the executable indices \(0,\ldots,947\) and
    \(0,\ldots,6507\), while the regime-specific pool and manifest hashes bind
    their complete serialized terms and lineage.
  - Construction begins from the deterministic ordered `full_meta` parent
    surface: lifted UCCSD, HVA, termwise Hamiltonian augmentation, pure-phonon
    generators, required PAOP families, Hamiltonian blocks, reusable
    fermionic generators, the registered optional PAOP and VLF/SQ families,
    and registered UCCSD--PAOP products. The source protocol records every
    ordered parent label; the builder source fixes each family loop and index
    range before parent-level deduplication.
  - Each parent is decomposed into one-Pauli atoms. The atom is accepted only
    after a fail-closed fixed-\((N_\uparrow,N_\downarrow)\) sector check. Its
    polynomial direction is then canonicalized to unit coefficient norm with
    a positive first nonzero anchor, making the identity invariant to an
    overall coefficient, sign, or complex phase. Global deduplication is by
    that projective Pauli identity while preserving all parent lineage; the HH
    binary-codeword hard filter is applied before Phase I; the identity/global-
    phase direction is removed; and the surviving order is deterministic.
  - The historical global-singleton materialization independently binds the
    same adapter identity, `global_guarded_singleton_pool_v1` Phase-I supply,
    all-executable visibility, identity-on-retained-singletons Phase-II
    exposure, and the six regime-specific pool hashes. The two lock-fixture
    tests were rerun separately at \(M=3\) and \(M=7\), with the pool cache
    disabled; both passed. These are method-definition and materialization
    proofs only. No promoted Bundle-5 trajectory used this flat supply, so a
    route-faithful rerun remains required before its numerical claims can be
    attached to the current Paper-I method.
- [x] **BENCH-01 — Hamiltonians and sectors.** **The benchmark physics is
  largely traceable, but the printed HH Hamiltonian omits an executed additive
  zero-point constant and the pure-Hubbard spin sector is under-specified.**
  - The promoted Bundle-5 HH records use \(L=2\), \(t=1\), open boundaries,
    blocked Jordan--Wigner fermion ordering, binary phonons, half filling, and
    the six Cartesian-product regimes
    \(U/t\in\{0.25,1.25,8\}\) and
    \(g/t\in\{0.353553\ldots,0.790569\ldots\}\) at
    \(\omega_0/t=1\). The recorded Hartree--Fock reference is the
    computational-basis state with one up- and one down-spin fermion and
    phonon vacuum. At half filling \(\bar n=N_e/L=1\), so the executed local
    coupling \(g(b_i+b_i^\dagger)(n_i-1)\) agrees with the displayed
    \(g(b_i+b_i^\dagger)(n_i-\bar n)\).
  - Both the frozen qubit Hamiltonian builder and the sector-ED builder honor
    `include_zero_point=true`, adding
    \(\tfrac12\omega_0 I\) for each of the two local modes. The displayed
    Hamiltonian instead prints \(\omega_0\sum_i b_i^\dagger b_i\), with no
    zero-point term. This is a literal manuscript--artifact mismatch, but it
    is only the constant \(L\omega_0/2\): it cannot change gradients,
    generator ranking, optimized circuits, or the reported same-cutoff
    \(\lvert E-E_{\rm ED}\rvert\), because the run and its ED comparator use
    the same shifted Hamiltonian. The manuscript should either print
    \(\omega_0\sum_i(b_i^\dagger b_i+1/2)\) or explicitly state that the
    conventional zero-point constant is included in all energies.
  - The noiseless pure-Hubbard benchmark uses \(L=3\), \(t=1\), \(U/t=1.25\),
    open boundaries, six blocked-order Jordan--Wigner qubits, no phonon
    register, and the fixed spin-resolved sector
    \((N_\uparrow,N_\downarrow)=(2,1)\). This is half filling but is the
    spin-imbalanced doublet sector; the manuscript currently says only
    “half-filled” and should give the pair for reproducibility. Its recorded
    reference is the six-qubit Hartree--Fock basis state `001011`, and the
    exact sector energy is \(-2.1414935642714075\,t\).
- [x] **BENCH-02 — Phonon notation.** The code and promoted records agree that
  \(L\) is the number of lattice sites and hence the number of local phonon
  modes, while \(M=n_{\rm ph,max}\) is the **inclusive** maximum occupation of
  each mode. Thus a local mode has dimension \(M+1\); \(M=3\) and \(M=7\)
  exactly occupy two and three binary qubits per mode. With four fermion
  qubits at \(L=2\), the recorded total register sizes are eight and ten
  qubits, respectively, and the Hartree--Fock reference appends the phonon
  vacuum in both cases. The plotted error compares each variational energy to
  sector ED at the identical \(M\), so “same-cutoff” is exact. It does **not**
  establish convergence of the truncated ED reference to the infinite-boson
  problem at the reported \(10^{-14}\)-scale variational errors; the source
  comments correctly keep the available cutoff scan diagnostic rather than
  promoting it as such a convergence claim.
- [x] **BENCH-03 — Inner optimizers.** **The compared HH prefixes use the
  common Powell tolerances stated in the manuscript.**
  - Bundle 5's frozen RA source defines the accepted-refit Powell settings as
    `xtol=1e-4`, `ftol=1e-8`, and `maxiter=200`.
  - The frozen AVQE source has native defaults `xtol=1e-5` and `ftol=1e-12`,
    but those defaults do not describe the promoted campaign. The
    authoritative rounds-1--50 tolerance overlay validates the source options
    at every Powell invocation and replaces only `xtol` and `ftol` with
    `1e-4/1e-8`; the rounds-51--75 continuation worker validates and applies
    the same overlay. No separate `maxfev` limit is introduced. Thus all 75
    AVQE rounds and the Bundle-5 RA rounds use the same reported Powell
    tolerances and `maxiter=200`. The earlier audit inference from the source
    defaults alone was wrong.
  - The local \(L=3\) optional-control campaign records a common Powell budget
    of `maxiter=200`, `xtol=1e-5`, `ftol=1e-12`; RA uses the supported
    Fubini--Study affine chart and AVQE uses native directions, as the Results
    section states.
  - Repository requirements pin SciPy 1.16.3 and the local environment reports
    that version. The CHTC runs fix a container-image hash but do not serialize
    a package-version inventory, so exact image-contained version closure
    remains a reproducibility receipt issue under REP-01. Powell callback
    semantics remain separately open under EST-08/QPU-04.
  - Manuscript consequence: no tolerance-asymmetry correction or rerun is
    required for the promoted HH comparison. Exact image-contained package
    versions and Powell callback accounting remain the separate open issues.
- [x] **BENCH-04 — Horizons and reporting.** The manuscript definition is
  reproducible: for each regime, take the first AVQE round \(k\leq60\) whose
  error is no more than \(10^{0.1}\) times the AVQE iteration-75 error, and use
  \(k^\ast=75\) if no such round exists. An independent calculation from the
  authenticated 75-round AVQE histories reproduces
  \(k^\ast=11,9,10,75,75,27\) in W--W, I--W, S--W, W--S, I--S, and S--S.
  - The weak row is a deliberate display crop through iteration 45; the
    underlying AVQE history extends to 75 and the promoted Bundle-5 RA
    histories extend to 50. The crop therefore does not imply that the weak
    RA runs terminated at 45.
  - The strong row has axes through 75 and plots every trajectory only through
    the prefix bound in the figure manifest. In the audited manifest, the
    Append/Plateau/Insert endpoints are \(75/57/51\) for W--S,
    \(75/60/48\) for I--S, and \(75/51/51\) for S--S; no shorter trace is
    extrapolated. These are a frozen figure snapshot, not a claim that the
    insertion routes ended or completed there.
  - In W--S and I--S, \(k^\ast=75\) is the prescribed fallback because no
    qualifying round occurs by 60. It is a reference round, **not** an
    empirically identified plateau-entry round. Any prose calling all six
    \(k^\ast\) values “the AVQE plateau point” is therefore inaccurate; the
    current table caption's “AVQE reference round” is the correct general
    term.
- [x] **BENCH-05 — Fixed-seed noisy diagnostics.** The six-cell campaign
  manifest closes three RA-Append and three AVQE trajectories at \(k=20\) for
  the half-filled six-qubit \(L=3\), \(U/t=1.25\) Hubbard problem and common
  24-generator pool. QPU-05 verifies the noisy/exact split, step 0.1, and seed
  tuple. The dotted lines are medians of each RA same-cutoff-error series over
  accepted \(k\ge10\), exactly as the figure provenance records; they are not
  analytic floors or ensemble statistics.

### G. Results, figures, tables, and prose claims

- [x] **RES-01 — Figure 2.** The six-panel display is mechanically traceable,
  but its caption identifies the disputed route as the current flat-pool RA.
  - The generator loads each RA trace in the declared deterministic order:
    promoted Bundle-5 table trace, transferred checkpoint archive, then live
    tail. It crops the three weak panels at 45, extends each strong axis to 75,
    leaves shorter strong traces unextended, and draws a method-specific
    marker at the AVQE \(k^\ast\) only when that trace reaches the round.
    The manifest records every displayed point, source path/member, source
    kind, SHA-256, point count, endpoint, and marker. The audited manifest and
    PDF hashes are `8389897e...afeef2` and `2bc3abad...3e4c3`.
  - The manifest contains AVQE plus all three Bundle-5 policies in each of the
    six regimes, and its frozen strong endpoints agree with BENCH-04. Thus the
    display itself is not missing a regime or silently extending a trajectory.
  - Scientific-authority caveat: the caption says that RA and AVQE use the
    single-Pauli-word pools. The AVQE trace does, but the Bundle-5 RA evidence
    executed the legacy parent-shortlist/Pauli-child compatibility route under
    POOL-01/EST-05. Therefore the data provenance is closed while the caption's
    method identity is false unless the hierarchy is reinstated as the method;
    otherwise the figure requires a route-correct rerun.
- [ ] **RES-02 — Table I.** **All circuit tuples and weak estimator prefixes
  are traced; strong AVQE callback work and route-authoritative RA evidence
  remain open.**
  - The six \(k^\ast\) values and every displayed AVQE/RA-Append circuit tuple
    reproduce the source tables. The nine weak insertion-policy circuit tuples
    likewise trace to exact accepted-prefix replay. The weak estimator
    convention closes to the values in EST-03. At three significant figures,
    the corrected \(10^3\)-scaled rows are
    \(13.8/6.27/8.73/17.8\),
    \(11.1/5.59/5.59/12.3\), and
    \(12.0/5.78/10.3/14.0\), ordered
    AVQE/RA-Append/RA-Plateau/RA-Insert. The present table instead retains the
    earlier candidate-only RA correction and AVQE `S_unique` values.
  - W--S and I--S have \(k^\ast=75\). The plotted plateau and always-insertion
    prefixes do not reach 75, so those dashes mean **no trajectory value exists
    at the reference round**. S--S has \(k^\ast=27\), and both insertion traces
    do reach that round; its dashes instead mean that their accepted-prefix
    circuit/estimator records have not been promoted into the table source.
    One glyph therefore currently represents two distinct states. The caption
    should distinguish “trajectory has not reached \(k^\ast\)” from “resource
    record not yet reported,” or the latter cells should be filled before
    submission.
  - Applying EST-07 to the three available strong RA-Append prefixes subtracts
    \(75/75/27\) outer-energy occurrences from the candidate-reuse-corrected
    totals \(389622/427501/42018\), yielding
    \(389547/427426/41991\). This changes the I--S rounded entry from 428 to
    427 (in units of \(10^3\)); W--S and S--S remain 390 and 42.0 at the shown
    precision. The matching strong AVQE values still require the full
    callback replay under EST-08, so no strong estimator percentage is frozen.
  - Scientific authority remains conditional on POOL-01: mechanically correct
    Bundle-5 table values do not make the legacy hierarchy representative of
    the current flat-singleton RA method.
- [ ] **RES-03 — Figure 3.** **The energy trajectories, circuit-resource
  matrices, branch/deletion annotations, and all circuit percentages are
  closed; estimator percentages remain provisional under EST-08.**
  - The RA curves and resource tuples trace to the affine-chart v10 campaign
    manifest and signed cell artifacts at source commit `8501ffd`; the AVQE
    curve is imported from the v9 provenance projection. The marked prefixes
    are AVQE \(k=12\), RA-Append \(k=12\), batch \(k=6\), beam \(k=12\), and
    ablation \(k=12\). The exact
    \(N_{2q}/D_{2q}/D_c\) tuples are respectively
    \(340/327/1345\), \(116/89/370\), \(212/192/798\),
    \(100/76/318\), and \(108/87/360\).
  - Those tuples exactly reproduce the printed mean RA reductions relative to
    AVQE: \(60.6\%\), \(66.1\%\), and \(65.7\%\) in
    \(N_{2q},D_{2q},D_c\), with the individual circuit reductions spanning
    \(37.6\%\)--\(76.8\%\). Relative to the common RA-Append prefix, batch
    makes a larger circuit; beam reduces the three circuit measures by
    \(13.8/14.6/14.1\%\); and ablation reduces them by
    \(6.9/2.2/2.7\%\). These circuit claims are frozen independently of the
    estimator recount.
  - The batch trajectory terminates after accepted round 11 on joint-rank
    exhaustion, and the plotted curve correctly stops there rather than being
    extrapolated to 15. The beam panel's dotted trajectories trace the
    executed children described in OPT-04, while the solid line and matrix
    follow the selected lineage. The ablation ticks occur at rounds 10--15;
    the \(k=12\) marker therefore follows three accepted delete--refit events.
  - The figure presently prints candidate-reuse-corrected estimator values
    \(7477/6065/4503/6340/7021\) for
    AVQE/RA-Append/batch/beam/ablation. Applying the already settled EST-07
    convention removes 12, 6, 12, and 12 outer-energy occurrences from the
    four RA prefixes, giving interim RA values
    \(6053/4497/6328/7009\). The AVQE source artifact used to make the v9
    projection is no longer present at its recorded private-temporary path;
    only the condensed error/cost provenance remains. Consequently its twelve
    outer refreshes can be identified in principle, but its memoized Powell
    callbacks cannot yet be restored under EST-08.
  - Therefore the printed \(20.0\%\) mean estimator reduction, the
    \(18.9\%\) RA-Append total reduction, the \(58.7\%\) Powell-call reduction,
    batch's \(25.8\%\) estimator reduction, beam's \(4.5\%\) premium, and
    ablation's \(15.8\%\) premium all reproduce the **current provisional
    numbers** but are not final QPU-faithful claims. Close them by recovering
    or rerunning the AVQE baseline with full Powell callback telemetry, then
    regenerate the matrix and all dependent percentages. No uncertainty in
    the circuit-resource conclusions depends on that recount.
- [ ] **RES-04 — Figure 4/Table II.** **Trajectories and circuit resources
  trace; estimator entries do not yet satisfy the final convention.**
  - The signed local campaign now contains completed 20-round AVQE and RA
    cells at all three perturbation levels. Its older source-map sentence that
    the AVQE cells were still mid-run is stale; the campaign manifest and six
    artifact receipts are the later authority for completion status.
  - The campaign manifest records terminal RA errors
    \(5.8180\times10^{-6}/2.5253\times10^{-3}/1.4823\times10^{-1}\) and AVQE
    errors \(2.4419\times10^{-5}/2.2078\times10^{-3}/4.6650\times10^{-1}\)
    for low/high/extreme. Figure 4 and the appendix trajectories use these
    fixed records through \(k=20\). At high perturbation,
    \(E_{\rm RA}/E_{\rm AVQE}=1.1438\): RA is 14.4% higher than AVQE, but AVQE
    is 12.6% lower than RA. The Discussion's statement that “AVQE's terminal
    error is 14.4% lower” uses the first denominator with the second wording and
    is quantitatively incorrect.
  - The iteration-10 circuit entries
    \(268/112,255/93,1049/380\),
    \(268/160,253/144,1044/591\), and
    \(181/42,153/29,604/124\) trace to the same accepted-prefix recompilation
    path used for the noiseless controls.
  - The printed RA \(\mathcal E\) entries 6.79/6.50/4.77 come from the earlier
    candidate-only reuse correction 6794/6497/4768. Applying EST-07 additionally
    removes ten `N_H_outer` refreshes, giving interim 6784/6487/4758 before
    rounding. The AVQE entries must likewise remove ten outer refreshes but may
    require adding memoized Powell callbacks under EST-08. Consequently neither
    side's displayed estimator values, nor the appendix terminal comparison
    30877 versus 21750, is frozen under the final convention.
  - The AVQE discrepancy cannot be repaired by changing only the displayed
    numbers. Exact-parameter cache hits also skipped the serial synthetic-noise
    oracle and its next value-noise draw. Under the manuscript's intended
    fresh-acquisition Powell convention, the three AVQE noise trajectories
    themselves need telemetry-complete no-cache replacement runs before the
    noise comparison can be considered execution-faithful.
- [x] **RES-05 — Abstract.** **The circuit, displayed-error, and
  fixed-realization noise statements reproduce the promoted data, but the HH
  method identity and estimator range do not survive the audit.**
  - From the current weak Table-I tuples, RA-Append's stated
    \(D_{2q}\) reduction range \(16.7\%\)--\(61.9\%\) and \(D_c\) range
    \(17.8\%\)--\(52.9\%\) are exact to the shown precision. At displayed
    iteration 45 every promoted RA trace is below AVQE in all three weak
    regimes; the closest case is S--W, where RA-Append is
    \(1.3805\times10^{-6}\) versus AVQE
    \(1.3967\times10^{-6}\). The \(L=3\) RA prefixes are all below
    \(10^{-13}\) with smaller compiled circuits, and the fixed-seed noise
    statement correctly names only low and extreme perturbations.
  - “Using the same operator pool as append-only ADAPT-VQE” is false for the
    promoted HH evidence under POOL-01: AVQE screens the flat singleton pool,
    while Bundle 5 executes the legacy parent-shortlist/child-expansion route.
    This clause requires a route-faithful rerun or an explicit redefinition of
    the evaluated RA pipeline; it cannot be repaired by wording alone while
    retaining the current-method claim.
  - Under the now-closed weak-prefix estimator convention, RA-Append uses
    \(54.5\%\), \(49.6\%\), and \(52.0\%\) fewer logical scalar estimators than
    AVQE in W--W/I--W/S--W, i.e. a \(49.6\%\)--\(54.5\%\) range and
    \(52.1\%\) mean. The printed \(48.7\%\)--\(53.6\%\) range comes from the
    superseded AVQE `S_unique` values and is wrong even before resolving the
    Bundle-5 route-authority conflict. Because any route-faithful rerun will
    replace these totals, the abstract estimator claim is not submission-ready.
  - The abstract appropriately attributes the \(L=3\) outcome to the complete
    RA pipeline rather than operator selection alone: RA and AVQE share the
    intact pool but use different selector and Powell-coordinate policies, so
    the present data cannot isolate those mechanisms.
- [x] **RES-06 — Results prose.** **All current weak-sector circuit and
  displayed-error arithmetic is correct; the estimator paragraph is not.**
  - Across the nine weak policy--regime comparisons, the printed mean
    \(D_{2q}/D_c\) reductions \(42.8/35.4\%\) reproduce Table I. The
    RA-Append means/ranges and its \(34.9\%\) mean \(N_{2q}\) reduction also
    reproduce the tuples exactly. The iteration-45 error statements agree
    with the figure projection, including always insertion's
    \(10^{-14}\)-scale W--W and I--W values.
  - Replacing the AVQE query totals by the callback-faithful EST-08 values and
    using the mathematically reused RA totals changes the weak query
    comparisons. RA-Append reductions are \(54.5/49.6/52.0\%\) (mean
    \(52.1\%\)); plateau reductions are \(36.7/49.6/14.6\%\) (mean
    \(33.6\%\)); and always-insertion premiums are
    \(28.9/11.3/16.4\%\) (mean \(18.9\%\)). The current prose's
    \(48.7\)--\(53.6\%\), \(13.1\)--\(48.7\%\), and
    \(13.3\)--\(31.3\%\) ranges are artifacts of earlier accounting.
  - These corrected values describe Bundle 5 as executed, not the intended
    current flat-singleton RA route. They belong in the audit record but must
    not be promoted as final method evidence unless the POOL-01 author
    decision retains that hierarchy.
- [x] **RES-07 — Discussion.** **Several interpretations are supportable, but
  the setup/claim paragraph currently combines four contradicted premises.**
  - MATH-11 proves that neither benchmark family activated the adaptive-width
    shortlist; the claimed common “\(\gamma=0.9\)” policy is false for all
    displayed runs. They used fixed widths \(8/4/1\) in Bundle 5 and normally
    \(24/12/1\) in the \(L=3\) controls. Adaptive width remains a method
    capability, not an evaluated setting.
  - For \(L=3\), attributing the difference jointly to operator selection and
    Fubini--Study-whitened refits is correctly cautious. No matched-coordinate
    control isolates whitening or selection. For HH, however, the closing
    phrase “singleton Hubbard--Holstein Pauli generators” misidentifies the
    Bundle-5 selection route under POOL-01, so the cross-setting mechanism
    claim must await route-faithful evidence.
  - The optional-control circuit interpretations and the concrete beam
    energy--cost example are supported by RES-03/OPT-04. Saying that up to
    twelve children *may* be executed in parallel is a scheduling capability;
    it is not evidence that twelve QPU partitions were actually used. The
    estimator percentages in the same paragraphs remain provisional for the
    reason recorded in RES-03.
  - At the high synthetic perturbation, RA/AVQE terminal error is 1.1438.
    Thus RA is \(14.4\%\) higher than AVQE, or equivalently AVQE is
    \(12.6\%\) lower than RA. The sentence “AVQE's terminal error is
    \(14.4\%\) lower” mixes the first denominator with the second wording and
    must be corrected.
  - The broad conclusion that the complete tested pipeline can improve
    accuracy and compiled circuit cost together is supported as a descriptive
    observation. It must remain bounded to these exact-state/single-realization
    benchmarks and must not be converted into a causal or universal hardware
    claim.
- [x] **RES-08 — Numerical precision.** The phrase is presently operationally
  undefined, although the underlying values are clear. At the four marked RA
  optional-control prefixes, the same-cutoff errors are
  \(1.20\times10^{-14}\) (Append), \(9.15\times10^{-14}\) (batch),
  \(4.88\times10^{-15}\) (beam), and \(2.22\times10^{-15}\) (ablation).
  The weak-HH insertion values described as converged likewise lie in the
  \(10^{-14}\)--\(10^{-16}\) range. These are differences between
  double-precision exact-state numerics and same-cutoff ED, not experimentally
  resolved physical errors.
  - Manuscript consequence: define one operational threshold—e.g.
    “within \(10^{-12}t\) of same-cutoff ED”—where the phrase is first used,
    or replace it with the actual error scale in each claim. Values below
    roughly \(10^{-13}\) should be grouped as numerical-floor convergence;
    their ordering must not be interpreted physically. The existing separate
    \(10^{-6}\) `accuracy_target` in the optional-control provenance is an
    early convergence diagnostic, not the meaning of “numerical precision,”
    and the two thresholds must not be conflated.

### H. Pseudocode, notation, and document consistency

- [x] **DOC-01 — Complete RA pseudocode.** **The branch/beam skeleton is
  recognizable, but the pseudocode is not presently a complete executable
  specification of either the printed method or the evaluated routes.**
  - Correct high-level structure: it fixes a branch-local population, applies
    Phases 0--III in order, obtains either singleton or batch proposals,
    creates refitted children, updates each child's radius, optionally trials
    ablation, Pareto/cost-ranks beam children, and returns the leading branch.
    The beam work is branch-local and the radius is updated before ablation,
    matching the audited implementation.
  - Missing record transition: the initial line constructs records directly
    from the already commutation-reduced set \(\mathcal P_k^u(A)\), but never
    maps each class to the positioned representative \(p^\star(A,[j]_A)\)
    before Phase I. That omission hides the distinction between the
    class-invariant Phase-0 response and the position-dependent resource
    record defined in the main text.
  - Missing policy evaluation: plateau triggering appears only as an input.
    The algorithm never evaluates Eq. 31 or shows how append-only, plateau,
    and always-insertion determine \(\mathcal P_k^u(A)\) in the current round.
  - Missing refit identity: “refit the complete ansatz” omits the zero versus
    Phase-III warm start and the fixed supported Fubini--Study affine chart.
    Because whitening is part of the manuscript's definition of the complete
    RA pipeline, this is not merely an implementation detail.
  - Incomplete exceptional paths: if a deletion trial is rejected, the
    pseudocode does not explicitly retain the already admission-refitted
    child; if one live branch has no feasible proposal, it provides no
    per-branch termination/removal rule before attempting to instantiate a
    child. The final global stopping line does not repair either local branch
    ambiguity.
  - Batch is underspecified relative to OPT-02: member order, cardinality/rank
    feasibility, the exact evaluated denominator, and the optional greedy
    alternative are absent. The beam skeleton is adequate at its declared
    abstraction level, but the ablation line does not expose the measured
    post-admission response acquisition established in OPT-03.
  - Evaluated-route mismatch: the `until` condition encodes the generic
    positive-Phase-III natural stop, while Bundle 5 uses forced horizons and
    the all-models-infeasible Gram-novelty fallback in MATH-12. Likewise the
    pseudocode always invokes the adaptive restriction even though the
    displayed evidence used fixed widths under MATH-11.
  - Manuscript consequence: the appendix can remain compact, but it must
    either be pseudocode for the general declared method—with the transitions
    and exceptional paths above—or be explicitly labeled a schematic. It
    cannot be called the “Complete RA algorithm” while silently mixing the
    adaptive generic method with unevaluated controls and evaluated-route
    exceptions.

- [x] **DOC-02 — Notation ledger.** **The core candidate-response notation is
  mostly coherent; five local collisions or redundant quotient operations
  should be resolved.**
  - Stable objects: \(A\) is a candidate generator, \(B_\ell\) an admitted
    active block, \(p\) an integer insertion position, \([j]_A\) its
    commutation class, \(p^\star\) the class's positioned representative, and
    \(r=(A,p)\) the positioned record. \(\boldsymbol\alpha_B\) and
    \(\delta\boldsymbol\theta\) preserve the candidate--active ordering of
    the singleton Phase-III chart. \(L\) is site/local-mode count, \(M\) the
    inclusive local phonon cutoff, \(k\) the outer iteration, and \(t\) the
    screening phase when used as a superscript. Context adequately separates
    that phase index from the hopping amplitude \(t\).
  - Redundant quotient: Eq. 3 already defines \(\mathcal P(A)\) as one
    representative per commutation class and therefore makes
    \(\mathcal R_j^{(t)}\) a quotient population. Appendix D then writes
    \(\mathcal R_j^{(t)}/\!\sim\) throughout the estimator formula. This is a
    second quotient of an already reduced set and conflicts with REC-01. Use
    the phase populations directly, or introduce a genuinely raw positioned
    population before quotienting; do not mix both conventions.
  - Energy-symbol overload: \(\Delta E(\mathbf z)\) is the signed quadratic
    change in Eq. 6, \(\Delta E_t\ge0\) is a phase predicted decrease, and
    \(\lvert\Delta E_k(M)\rvert\) is the reported same-cutoff physical error.
    The signs are audited as correct in MATH-02, but one symbol denotes three
    different typed quantities. A distinct signed-model symbol is the
    smallest cleanup.
  - \(E_0\) denotes the local source energy in Eq. 5 and later the common
    initial energy of an entire beam in Eq. 35. The latter should use a
    beam-specific initial-energy symbol; these are conceptually different
    anchors even when both are constants in their local formulas.
  - The singleton factors \(K_t(r)\) are defined, but \(K_3(B)\), \(K_p\),
    and \(K_b\) are only described verbally. OPT-02 proves that the evaluated
    batch denominator is \(1+\sum_{r\in B}\kappa_2(r)\), not an unspecified
    population-normalized Phase-III factor; OPT-03/04 similarly require the
    exact deletion and beam comparison populations to make their relative
    factors reproducible.
  - The displayed ablation term \(g(p)\beta_p\) denotes a
    gradient-corrected Schur loss, whereas the plotted route executes
    `stationary_gw_zero_v1` and omits that gradient from the right-hand side.
    This is a formula/evidence mismatch already established in OPT-03, not a
    harmless notation choice.
  - \(W\) as the whitening map and \(W_b\) as beam compiled burden are
    distinguishable by subscript but unnecessarily dense. More importantly,
    “equivalence classes of the columns of \(W\)” in the dormant whitening
    derivation is not well typed: the quotient classes are circuit-coordinate
    displacements modulo the Gram kernel, represented by the columns, not
    equivalence classes *of columns*.

- [x] **DOC-03 — Terminology.** **A fixed vocabulary is available, but the
  current manuscript mixes method names and overstates two policy verbs.**
  - `ADAPT-VQE` is the proper first-use name and `AVQE` is introduced once at
    the start of Results. The abstract correctly avoids the shorthand. In the
    methods, “standard ADAPT” can denote the gradient rule, but `AVQE` should
    remain the results-series label rather than alternate with bare `ADAPT`
    for the same plotted comparator.
  - Use the route names **RA-Append**, **RA-Plateau**, and **RA-Insert** in
    displays and quantitative comparisons, mapping them once to append-only,
    plateau-triggered, and always-open insertion. The present prose alternates
    among “always insertion,” “always-insertion,” and `RA-Insert`; this is
    readable but not publication-consistent.
  - `position` is the correct selectable object and `factor`/`factorization`
    the correct circuit split; active text no longer needs `cut`. Use
    `equivalence class` throughout. `response class` is absent and should not
    be reintroduced.
  - Use **optional controls** as the umbrella, with **batch admission**,
    **generator ablation**, and **beam continuation** as the named controls.
    The source currently mixes “Optional RA algorithm variants,” “optional
    structural controls,” “optional-control settings,” and `RA-Extension`.
    `RA-Extension` is acceptable only as the common visual legend role, not as
    a fourth algorithmic category in prose.
  - “Always-insertion admits from every class” is logically false: the policy
    permits candidates from every class and admits only the winning record in
    a singleton round. The TETRIS sentence likewise should say its joint layer
    *requires* disjoint supports, while RA batch does not impose that
    particular gate; it must not imply absence of all feasibility gates.
  - Reader-facing prose consistently uses `ansätze`; bibliography titles
    preserve their published spelling and require no normalization.

- [x] **DOC-04 — Internal duplication.** **The audit separates useful
  restatement from contradictory repetition and release-only source clutter.**
  - Useful restatement: the main Results defines exact-state/noise scope,
    error, compiler tuple, and horizons before the displays; Appendices C--F
    then give pool, estimator, noise, and cost details. These layers serve
    different readers and need not be collapsed merely because a symbol
    recurs.
  - Contradictory repetition: the Discussion restates the benchmark controls
    as adaptive \(\gamma=0.9\), common \(24/24/12/12\) caps, three-weight
    compiled selector cost, and ridge-free Gram support. MATH-11 and
    COST-CORE-01/02 show that the displayed HH evidence used fixed widths and
    a different no-transpile five-coordinate oracle; MATH-09 shows a
    \(10^{-9}\) retained-support whitening ridge. This paragraph is not a
    harmless summary—it consolidates false evaluated settings and should be
    replaced rather than duplicated elsewhere.
  - Results/Discussion boundary: pages 10--12 already give the trajectories,
    matrices, and several exact comparisons, while the section titled
    “Discussion and conclusion” opens with another parameter ledger and then
    repeats per-control numerical reductions. The quantitative trend prose is
    useful, but settings belong in the benchmark protocol and the closing
    section needs synthesis, limitations, and the scoped contribution rather
    than a second results dump.
  - The main noise plot and Appendix-E plot show the same six trajectories
    through \(k=20\). The appendix version adds terminal annotations and the
    detailed diagnostic interpretation, so it is explanatory redundancy, not
    a contradiction; if page economy is required, the second full trajectory
    plot is the obvious candidate for removal while retaining the noise model
    and terminal diagnostic text.
  - The source retains an entire obsolete whitening derivation inside
    `\iffalse`, including a duplicate `eq:accepted_refit_whitening` label,
    as well as unused red-placeholder macros and extensive private absolute
    paths/hashes in comments. None renders in the PDF. They may remain during
    coauthoring, but a released source should move historical derivations and
    local provenance to a separate archive/manifest and delete unused
    editorial macros. REP-04 governs whether any such source release occurs;
    this audit does not presume one.

- [x] **DOC-05 — Cross-references, ordering, and rendered-page audit.** **The
  compiled snapshot is reference-clean, but four page-flow/source-release
  defects remain.**
  - Evidence: the current `Paper_I.log` contains no undefined reference,
    undefined citation, multiply-defined label, or overfull-box warning.
    Source inspection finds no reference target absent from the source, and
    the current 19-page PDF opens cleanly with all equations, figures, tables,
    citations, and appendix headings rendered. The dormant duplicate
    whitening label is skipped by TeX only because it lies inside
    `\iffalse`; it remains a source-maintenance hazard rather than a current
    build error.
  - Appendix ordering is internally correct: A gives numerical scoring, B the
    pseudocode, C pools, D estimator accounting, E noise, and F the
    transpiler-free estimates. The appendix begins on a new page, and the
    transpiler-free section is last as intended. Figure 3's control reference,
    Figure 4's noise reference, and the pool/estimator references resolve to
    their actual sections.
  - Page 9 leaves roughly half a page blank before the full-width Figure 2;
    page 11 uses only its upper half for Figure 3; and page 12 places the
    compact noise figure/table in the left column while forcing the entire
    Discussion and conclusion into a dense right column. These are visible
    page-flow defects even though no box overflows. The Results/Discussion
    restructuring in DOC-04 should be judged against these pages rather than
    by source line count alone.
  - The widetext pseudocode ends on page 14 with a conspicuous horizontal-rule
    remnant above the right column before Appendix C. The appendix noise plot
    on page 17 contains resource annotations and legend text too small for
    comfortable one-column reading. Both require visual recheck after the
    eventual source edit.
  - PDF metadata reports no title, author, subject, or keywords even though
    title/authors render on page 1. This does not affect mathematical content,
    but the final submission build should populate ordinary PDF metadata and
    replace `\date{\today}` by the intended submission convention.

### I. Reproducibility and evidentiary boundaries

- [x] **REP-01 — Software and compiler environment.** **The required stack is
  identifiable, but the Bundle-5 execution image lacks a package-level runtime
  receipt.**
  - The repository requirements pin NumPy 2.3.5, SciPy 1.16.3, Matplotlib
    3.10.7, Qiskit 2.3.1, and Qiskit Aer 0.17.2. The \(L=3\) accepted-run
    compile receipts independently record Qiskit 2.3.1 and a
    `FakeMarrakesh` target snapshot with backend version 1.0.7; the Results
    protocol records SciPy 1.16.3 for Powell. The plotting scripts use the
    noninteractive Matplotlib `Agg` backend.
  - Bundle 5 freezes its application source and execution-image hashes, and
    the repository requirement pins match the current local environment, but
    its archive does not serialize `python --version` or a `pip freeze` from
    inside the CHTC image. Consequently the exact Python/NumPy/SciPy/Qiskit
    versions *inside that image* are not proved by the result archive. A
    requirements file is intent, not a runtime receipt.
  - Reproducibility consequence: retain the source/image hashes already bound
    to each campaign, and for any route-authoritative rerun emit a compact
    environment receipt containing Python and the five package versions above,
    the Qiskit fake-backend snapshot, platform/architecture, and optimizer
    implementation. The manuscript need not print a package dump, but its
    eventual availability artifact should bind the exact environment used for
    every promoted benchmark family.
- [x] **REP-02 — Randomness.** **All stochastic inputs used by the displayed
  evidence are identified, and the noiseless optimizations are deterministic.**
  - The exact-state HH and noiseless \(L=3\) calculations use deterministic
    Powell refits. Their route records carry algorithm seed 7, but no sampled
    optimizer step is active. All post-transpilation circuit resources use
    `seed_transpiler=7`; the \(L=3\) compile receipt confirms that seed and the
    exact fake-backend target.
  - The fixed-realization noise diagnostics bind value-noise seed 702688422,
    coherent-overrotation seed 20260609, simulator seed 7, and transpiler seed
    7. These values are common to the paired RA/AVQE construction at each
    perturbation strength. QPU-05 separately records which quantities are
    noisy and which remain exact.
  - No displayed uncertainty band is a seed ensemble: the noiseless results
    are deterministic, and each noise level is one frozen realization. Thus no
    additional “optimizer seed” or statistical confidence interval should be
    inferred. A future ensemble claim would require a declared seed list and
    aggregation rule rather than reusing this single-realization protocol.
- [x] **REP-03 — Source maps.** **Every present figure and table has a local
  source map, but release portability remains a separate availability task.**
  - Figure 1 is an inline schematic and names its inline LaTeX generator.
    Figures 2--4 name both a provenance manifest and a plotting script;
    Tables I--II name the underlying table source or in-script data block and
    identify LaTeX as the renderer. The appendix noise figure likewise records
    its six run artifacts, plotting script, estimator-correction receipt, and
    regeneration command, although it predates the standardized
    `DISPLAY_PROVENANCE` label. Every named file and script exists in the
    audited local checkout.
  - The manifests preserve source hashes, plotted prefixes, and output hashes.
    Figure 2 is especially complete: every series records its source
    path/member, source kind, SHA-256, endpoint, and marker status. Figure 3
    retains the v9 AVQE projection manifest rather than the now-missing raw
    temporary run directory; that is enough to trace the plotted energy and
    circuit values, but not enough to reconstruct the raw Powell callback
    telemetry required by EST-08.
  - The comments do not render in the paper, so their absolute `/Users/...`
    paths are not reader-facing prose. They are nevertheless machine-specific:
    the figure generators and raw data live in sibling private worktrees rather
    than in the manuscript tree. Before any reproducibility release, replace
    these paths by release-relative identifiers and package the referenced
    scripts/manifests/data according to the author decision in REP-04. This is
    a packaging issue, not an untraced-display issue.
- [x] **REP-04 — Availability.** **No public-release commitment is present or
  authorized; availability is retained as an explicit author decision.**
  - The manuscript contains no Code Availability or Data Availability
    statement and names no public repository, DOI, archive, or upon-request
    channel. The local git repositories, sibling worktrees, plotting scripts,
    run archives, and JSON manifests are therefore audit evidence, not
    implicitly public supporting information. In particular, a machine-readable
    pool builder or JSON provenance file need not be embedded in the article,
    and this audit must not convert the existence of local code into a promise
    to open-source it.
  - The material that *could* support a future release is already separable:
    manuscript source and figure binaries; processed figure/table manifests;
    plotting and table-generation scripts; accepted-prefix result payloads and
    estimator receipts; and the larger raw checkpoint/archive layer. The
    present source maps in REP-03 identify these layers locally but do not make
    them externally retrievable.
  - Before submission, the authors must choose one of the journal-compatible
    boundaries—public archive, controlled/upon-request access, or a justified
    restriction—and write a conventional statement matching that choice. Any
    public option must replace local absolute paths with stable release-relative
    identifiers and should name the exact release commit or archive version.
    Until that decision is made, the manuscript should make no availability
    claim and the audit will not presume one.
- [x] **REP-05 — Claim class.** **The manuscript's material statements divide
  into four evidence classes; the audit identifies the places where one class
  is presently being used as evidence for another.**
  - **Mathematical design:** the four response models, trust-region solve,
    commutation quotient, robust relative-cost factor, whitening chart,
    admission/refit map, and optional-control definitions. These require
    internal derivational consistency and implementability, not benchmark
    success. Sections C--E audit them. A capability such as adaptive
    shortlisting, optional position insertion, or twelve-child beam generation
    remains a design claim unless a promoted run receipt activates it.
  - **Author-data result:** every trajectory, resource tuple, error value,
    percentage, plateau/reference round, and noise ordering. These require a
    bound artifact and a named comparator/denominator. Sections F--G audit
    them. A completed calculation cannot establish that it represents the
    current method when its route contract contradicts the method definition;
    the Bundle-5 pool conflict is the central example.
  - **Implementation fact:** pool construction, phase widths, optimizer chart
    and tolerances, compilation target/seed, estimator-ledger semantics,
    noise surfaces/seeds, and branch accounting. These require frozen source
    or run receipts. They must not be inferred from a displayed equation or
    repository default. Conversely, code-level recomputation that is
    mathematically reusable does not by itself justify charging duplicate
    logical estimators.
  - **Literature-backed claim:** descriptions of ADAPT, TETRIS, pruning,
    geometric/natural-gradient methods, measurement grouping, and statements
    about what prior methods do or do not provide. These require a primary
    citation whose scope matches the sentence; REP-06 audits the boundaries.
  - Current cross-class errors are now explicit elsewhere in the ledger:
    adaptive \(\gamma=0.9\) is a method capability presented as an evaluated
    setting (MATH-11/RES-07); the flat-singleton RA definition is attached to
    legacy hierarchical data (POOL-01/RES-01); logical scalar work is described
    as literal QPU acquisition (QPU-02); branch-cap concurrency is presented as
    observed hardware parallelism (QPU-08); and fixed-realization noise is at
    risk of being read as a general robustness claim (QPU-05). These are
    evidence-class mismatches, not stylistic preferences.
- [x] **REP-06 — Citation boundaries.** **Primary-source review supports the
  baseline, CEO, Geo/Pos-Geo, Hessian-recycling, TETRIS, and pruning
  distinctions, but exposes two false method descriptions and over-broad
  absence claims.**
  - Standard ADAPT's largest-gradient append rule is accurately attributed to
    Grimsley et al.; qubit- and qubit-excitation variants are described at the
    appropriate level. CEO-ADAPT does couple related qubit excitations and
    combines circuit/measurement-saving subroutines. Geo-ADAPT applies a
    pool-space Fubini--Study/natural-gradient selection rule, and its paper
    explicitly defines Pos-Geo-ADAPT as allowing beginning, interior, or tail
    insertion. Hessian recycling carries approximate second-order optimizer
    information across adaptive iterations to reduce optimization work.
  - The sentence “Overlap-ADAPT incorporates tangent-direction information
    into the append criterion” is false. The cited method grows an ansatz by
    maximizing overlap with an intermediate target wavefunction and then uses
    that ansatz to initialize ADAPT; it is not a tangent-Gram selection rule.
    See Feniou et al., *Communications Physics* **6**, 192 (2023):
    <https://www.nature.com/articles/s42005-023-01312-y>.
  - The sentence that “other methods exploit second-order derivative
    information to improve convergence” is not supported by the cited
    Grimsley 2023 paper. That source analyzes parameter recycling, local
    minima, trap burrowing, and barren-plateau avoidance; it does not propose
    Hessian-based candidate selection. See Grimsley et al., *npj Quantum
    Information* **9**, 19 (2023):
    <https://www.nature.com/articles/s41534-023-00681-0>. The adjacent
    Hessian-recycling citation supports only the narrower optimizer-recycling
    statement.
  - The TETRIS contrast is correct: its batch rule admits multiple operators
    only with disjoint qubit supports. The RA joint batch does not need that
    gate. See TETRIS-ADAPT-VQE:
    <https://arxiv.org/abs/2209.10562>. The Pruned-ADAPT citation and page
    range 8720--8728 are correct, but its rule is specifically
    amplitude-and-position based with a dynamic threshold; it deletes without
    immediate parameter reoptimization. RA's measured delete--refit gate is a
    distinct mechanism and must not inherit Pruned-ADAPT's “cost-free”
    characterization. See Vaquero-Sabater et al., *JCTC* **21**, 8720--8728:
    <https://pubs.acs.org/doi/10.1021/acs.jctc.5c00535>.
  - Broad claims that existing methods do not estimate candidate energy gain
    or do not use hardware/resource information are untenable as written.
    ExcitationSolve-ADAPT reconstructs each candidate's one-parameter energy
    landscape and selects the strongest immediate energy decrease; a 2024
    Hardware-Aware ADAPT experiment multiplies the gradient by a
    connectivity-derived hardware factor. A June-2026 energy-based ADAPT
    preprint also reduces the cost of per-candidate energy reconstruction.
    These do not duplicate RA's joint candidate--active Schur response, staged
    measurement funnel, marginal compiled-prefix competition, and positional
    equivalence classes, but they rule out the unqualified broader claims.
    Primary sources: ExcitationSolve-ADAPT
    <https://www.nature.com/articles/s42005-025-02375-9>, Hardware-Aware ADAPT
    <https://www.nature.com/articles/s41534-024-00952-4>, and
    resource-efficient energy-based ADAPT
    <https://arxiv.org/abs/2606.04786>.
  - The defensible novelty boundary is therefore: “To our knowledge, among
    the methods cited here, none combines candidate-specific joint
    candidate--active second-order response, staged measurement restriction,
    marginal compiled-resource normalization, and commutation-reduced
    insertion-position selection.” The analogous claim that no broadly adopted
    fermion--phonon ADAPT construction exists should likewise be framed as a
    scoped literature assessment, not an absolute fact. The vibrational and
    continuous-variable bosonic citations do support the statement that ADAPT
    has recently extended beyond purely fermionic molecular benchmarks.

## Current blocking boundary

The seven unchecked entries are not seven unaudited scientific topics. They
collapse to two evidence dependencies that the local archive cannot supply:

1. **Route-authoritative HH evidence:** EST-03, EST-06, and RES-02 require an
   HH campaign that actually executes the current measured-residual,
   flat-singleton RA method with the audited marginal compiled-cost contract.
   POOL-04 proves that the current RA and AVQE facades expose identical ordered
   singleton populations, but the exhaustive packaged-protocol scan under
   EST-06 proves that no existing tracked HH package executed that combination.
   Bundle 5 and Bundle 6 cannot be relabeled into compliance because their
   parent-shortlist/child-expansion authority is explicit in the contracts and
   receipts.
2. **Raw AVQE Powell callback telemetry:** EST-03, EST-08, QPU-04, RES-03,
   and RES-04 require the callback count that `append.py` discarded in favor of
   its exact-parameter cache-miss count. The three weak HH reference prefixes
   were recoverable by frozen-source replay, but the remaining source artifacts
   either omit the raw callback trace or no longer exist. For the noise panels,
   cache hits also suppress oracle calls and value-noise draws, so the missing
   evidence cannot be repaired arithmetically; a no-cache, telemetry-complete
   comparator replacement is required.

No reader-facing prose edit can close either dependency. The audit has frozen
the applicable mathematical accounting convention and the acceptance criteria
for replacement evidence; manuscript changes remain deliberately deferred.

## Immediate execution order

1. Close EST-08/QPU-04 by preserving or reproducing raw AVQE Powell callback
   counts for the remaining displayed prefixes. The replacement route should
   serialize both raw SciPy callbacks and distinct executed acquisitions;
   freeze no final estimator comparison before that closure.
2. Resolve the route-authority cluster exposed by POOL-01,
   COST-CORE-01/02, and MATH-11/12: flat singleton population, selector cost
   oracle, shortlist/stopping policy, and matched optimizer protocol must
   belong to one evidence route.
3. With the route-authoritative 948/6508 singleton inventories frozen under
   POOL-04, apply the closed estimator convention under EST-03/06 and
   regenerate the affected HH displays and percentages from the replacement
   route.
4. After evidence replacement, rerun the literal figure/table/PDF audit and
   resolve the manuscript consequences recorded in RES-02--04 and DOC-01--05.

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
