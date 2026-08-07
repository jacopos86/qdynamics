# Formal-Manifold Query-Closure Phase-Model Handoff

**Date:** 2026-07-11  
**Target:** the main implementation agent working on the formal-manifold route  
**Action:** implement query-closed Phase-I, Phase-II, and batch/Schur models  
**Only side-conversation deliverable:** this handoff  
**Do not:** edit the manuscript, launch scientific runs, change canonical
defaults, or count classical matrix algebra as quantum-estimator work

## Binding user clarification

For this task, **shots** means **quantum-estimator oracle queries**. The cost
unit is the set of unique logical estimator primitives requested. It is not
physical sample count, wall time, candidate count, matrix-element count, or the
number of classical contractions applied to one oracle response.

The user wants the formal-manifold version to exhaust the algebraic closure of
information already returned by those queries:

1. strengthen Phase I when its already-charged candidate query plus the
   retained manifold state determine the candidate-specific augmented Gram;
2. make Phase II consume that same record without reconstructing or requerying;
3. let batch/Schur scoring assemble every available joint block from retained
   tangent data before requesting a missing primitive;
4. pass the accepted candidate or batch record directly into manifold growth;
5. prove every claimed zero-query improvement through primitive identity.

The Phase-I closure applies uniformly to every candidate-position record:

- parent macro generators in their actual execution/parameterization mode;
- Pauli-child records produced by child-set expansion;
- singleton and batch candidates.

A macro tangent and its child tangents may be related algebraically, but they
are not interchangeable query identities unless the provider explicitly
returns a decomposition that proves that closure.

This is a scoped amendment to the earlier instruction that the first
formal-manifold implementation must leave selector equations unchanged. The
**experimental formal-manifold composition only** may use the query-closed
models below. Existing routes and canonical defaults remain unchanged. The
enrichment must not add estimator-oracle primitives beyond the existing Phase-I
and Phase-II query frontier, but it must consume second-order candidate data
already returned by the established Phase-II oracle contract.

## Governing sources

Read in order:

1. `AGENTS.md`
2. `MATH/AGENTS.md`
3. `MATH/paper_facing/README.md`
4. `MATH/paper_facing/paper_I_static_scaffold/paper_i_formal_manifold_warm_start_20260711.md`
5. `MATH/paper_facing/paper_I_static_scaffold/paper_i_integrated_geometric_recycling_implementation_contract_20260711.md`
6. `MATH/paper_facing/paper_I_static_scaffold/paper_i_integrated_geometric_recycling_implementation_handoff_20260711.md`
7. this handoff
8. the current source and tests named below

The formal note remains authoritative for metric, curvature, transport, rank,
and rollback semantics. This handoff controls the later user-requested
query-closure and selector-model integration.

Do not edit `MATH/paper_details/Paper_I.tex` or rebuild `Paper_I.pdf`.

## Verified checkout snapshot

At handoff creation:

```text
cwd: /Users/jakestrobel/local_repos/Holstein_test_fullclone_3
branch: codex/local-repo-health-20260706
HEAD: e1478cfbd6aa1792cd271c1c4f39b4d7f64ce8e8
```

Reverify before editing. Relevant paths are already dirty or untracked:

```text
M  pipelines/scaffold/hh_continuation_scoring.py
M  pipelines/scaffold/hh_continuation_types.py
M  pipelines/static_adapt/adapt_candidate_record_cache.py
M  pipelines/static_adapt/adapt_pipeline.py
M  pipelines/static_adapt/selector_measurement_proxy.py
M  test/test_hh_continuation_scoring.py
M  test/test_static_adapt_controller_measurement_work.py
?? pipelines/static_adapt/formal_manifold_exact_backend.py
?? pipelines/static_adapt/formal_manifold_warm_start.py
?? pipelines/static_adapt/route_a_funnel.py
?? pipelines/static_adapt/route_a_schur_selector.py
?? test/test_static_adapt_route_a_funnel.py
```

Treat every existing hunk as intentional user work. Use path-limited status
and diffs. Do not reset, revert, overwrite, stage, commit, or switch branches.

## Verified current-code audit

The main agent must extend existing machinery rather than create a parallel
selector stack.

### Phase I already materializes a candidate tangent

The Phase-I/full-record helpers in
`pipelines/static_adapt/adapt_pipeline.py` currently compute:

```text
apsi_candidate
mean_candidate
centered_candidate
grad_candidate
metric_candidate = Re(centered_candidate^dagger centered_candidate)
```

The `evaluation_stage_key == "phase1"` path stores scalar gradient and metric
fields in `CandidateFeatures`, then discards the live `centered_candidate`.
This is the first concrete reuse gap. On the exact-state backend, retaining a
round-local typed tangent handle makes overlaps with the retained active frame
classical contractions. Apply the same typed path to macros and Pauli children;
do not leave parent macros on a weaker scalar-only Phase-I model.

Relevant surfaces:

```text
pipelines/static_adapt/adapt_pipeline.py
  _full_record_for_candidate_local(...)
  _full_record_for_candidate(...)
  evaluation_stage_key == "phase1"

pipelines/scaffold/hh_continuation_scoring.py
  _tangent_data(...)
  build_candidate_features(...)
```

### Phase II already has a strong but late reuse payload

`pipelines/scaffold/hh_continuation_scoring.py` already contains:

```text
_exact_insertion_joint_geometry_payload(...)
_build_phase2_joint_geometry_cache(...)
_phase2_joint_geometry_reuse_accounting(...)
evaluate_phase2_joint_response_singletons(...)
```

`phase2_joint_geometry_reuse_v2` includes state/chart fingerprints and:

```text
G_AA, G_AB, G_BB
H_AA, H_AB, H_BB
g_A, descent_gradient
```

Its cache validates state, ordered scaffold, theta, Hamiltonian, candidate
coordinate, insertion position, active context, and reconstruction tolerance.
Preserve those fail-closed checks.

Current limitations:

1. the rich payload is produced only after the Phase-I shortlist;
2. some paths still emit the weaker v1 append chart;
3. Gram, ordinary energy Hessian, and differential data share one loose payload;
4. ordinary Hessian data must never be relabeled covariant curvature or the
   optimizer's inverse-curvature model;
5. candidate Hessian blocks are not query-free merely because Gram blocks are;
6. when the existing Phase-II response already returns candidate ordinary
   second derivatives, the new route must consume them rather than discard them.

### Batch/Schur already reuses singleton blocks

The same scoring module contains:

```text
_build_batch_full_geometry_workspace(...)
_required_joint_candidate_pairs(...)
_measure_joint_candidate_pair_entry(...)
select_phase2_batch_record_proposals(...)
```

It imports validated singleton blocks, constructs required candidate pairs,
caches pair results, and exposes reuse telemetry. Preserve its state and
candidate-identity gates.

When all candidate tangent handles coexist in one round-local workspace,
candidate-pair Gram entries are already classical contractions. In that
capability mode they must not pass through a query-charging pair path. Mixed
candidate objective-Hessian entries remain a separate information class.

### Query accounting currently mixes two currencies

Relevant code:

```text
pipelines/static_adapt/route_a_funnel.py
  RouteAFunnelQueryEvent
  RouteAFunnelResult.query_work

pipelines/static_adapt/selector_measurement_proxy.py
  OPERATOR_PROBE_CHARGE_BASIS
  record_joint_selector_workspace_work(...)
  ControllerMeasurementWorkAccumulator
```

The controller ledger uses typed logical operator probes with charge basis
`logical_estimator_request_pre_grouping_v1`. The joint workspace also reports
`query_chargeable_unique_geometry_element_count`, which can currently be
converted directly into `actual_operator_probe_count`.

Do not assume one matrix entry always equals one estimator-oracle query. The
provider must declare the primitive-to-output map. Retain matrix-element counts
as diagnostics and charge unique logical estimator primitive IDs. If the
established provider maps one entry to one query, reconciliation will prove the
counts equal.

## Required oracle-query identity

Define a query by an immutable primitive identity containing at least:

```text
primitive kind:
  energy | coordinate_gradient | tangent_or_metric |
  coordinate_second_derivative | hessian_vector | cross_state_tangent
physical-state fingerprint
beam-branch identity
ordered scaffold fingerprint
theta fingerprint
logical-coordinate registry fingerprint
candidate generator fingerprint
candidate insertion position
parameterization/tie-map fingerprint
Hamiltonian fingerprint when relevant
provider/backend identity
requested estimator/precision contract
logical observable/formula primitive identity
```

Reuse is legal only for the same identity. Different states, branches,
insertions, tie maps, Hamiltonians, or estimator contracts are different
queries.

Every provider response should expose a receipt:

```text
QueryReceipt
  schema
  primitive_ids_requested
  primitive_ids_reused
  returned_fields
  closure_capabilities
  provenance_by_field
  provider_kind
  statevector_shortcut_used
```

For zero-increment fixtures, acceptance requires set equality:

```text
new_unique_primitive_ids(enriched route)
  == new_unique_primitive_ids(baseline route)
```

Final scalar count equality alone is insufficient.

## Required typed query-closure workspace

Introduce or extend a narrow typed workspace. Names may follow local style.

### Current-manifold anchor

```text
SelectorGeometryAnchor
  state_fingerprint
  branch_id
  manifold_id
  ordered_scaffold_fingerprint
  theta_fingerprint
  coordinate_registry_fingerprint
  parameterization_mode
  active_coordinate_indices
  active_tangent_frame_or_handle
  G_AA
  b_A
  gram_provenance
  differential_provenance
  source_query_receipts
```

For the formal route, obtain this read-only anchor from the accepted manifold
state or exact endpoint evaluation. Do not recompute it in each phase. If the
stored metric is predicted, preserve that provenance; reuse does not make it
exact.

### Candidate response

```text
CandidateTangentRecord
  candidate identity and insertion position
  candidate coordinate registry entry
  state/chart fingerprints
  tangent vector or provider tangent handle
  b_B
  G_AB
  G_BB
  query receipts
  closure capabilities
  provenance by block
```

The tangent/handle is round-local and need not be serialized. Portable records
retain numeric blocks, fingerprints, receipt IDs, and provenance. Do not write
dense statevectors or provider handles into the disk candidate cache.

### Population workspace

```text
QueryClosedPopulationWorkspace
  shared anchor
  ordered candidate records
  G_AC
  G_CC when closure permits
  b_C
  missing primitive requests
  unique primitive ID set
  derived-feature cache
  subset-solve cache
```

Phase I, Phase II, batching, and accepted growth consume views of this same
state/branch-scoped workspace.

## Mathematical query-free closure

Let `A` denote retained ansatz coordinates and let candidate `mu` be inserted
at zero angle:

\[
G_\mu^+
=
\begin{bmatrix}
G_{AA} & g_{A\mu}\\
g_{A\mu}^{\mathsf T} & g_{\mu\mu}
\end{bmatrix},
\qquad
b_\mu^+
=
\begin{bmatrix}
b_A\\ b_\mu
\end{bmatrix}.
\]

When the existing response supplies the candidate tangent and old frame, this
requires no new query. Define:

\[
s_\mu
=g_{\mu\mu}-g_{\mu A}G_{AA}^{\dagger}g_{A\mu},
\qquad
\widetilde b_\mu
=b_\mu-g_{\mu A}G_{AA}^{\dagger}b_A.
\]

For a candidate block `B`:

\[
S_B
=G_{BB}-G_{BA}G_{AA}^{\dagger}G_{AB},
\qquad
\widetilde b_B
=b_B-G_{BA}G_{AA}^{\dagger}b_A.
\]

Once these blocks have been returned, the following are classical and free:

- augmented Gram construction;
- residual Schur Gram;
- retained spectrum, rank gain, spectral gap, and condition estimate;
- old-span projection coefficients and novelty fraction;
- natural-gradient response and trust-radius first-order reduction;
- candidate-pair Gram contractions from live tangent handles;
- subset ranking, permutation, and Schur solves;
- growth factorization and projector enlargement.

## Required Phase-I model

For the experimental formal composition, Phase I should use the strongest
first-order model determined by the already charged query and retained anchor.
For a rank-feasible singleton:

\[
R_\mu
=\frac{\widetilde b_\mu^2}{s_\mu+\lambda_G},
\qquad
\Delta E_{1,\mu}^{\rm qc}
=\rho\sqrt{\max(R_\mu,0)}.
\]

This is the optimum of the linearized objective along the residualized
candidate direction under the metric trust-radius constraint. Retain the
existing resource denominator:

\[
S_{1,\mu}^{\rm qc}
=\frac{\Delta E_{1,\mu}^{\rm qc}}{1+K_{1,\mu}}.
\]

Requirements:

1. retain legacy Phase-I fields for compatibility and telemetry;
2. add explicitly named query-closed fields;
3. make the new score authoritative only in the experimental composition;
4. rank-gate below the declared threshold;
5. retain block-level provenance;
6. declare zero incremental queries only after primitive-ID reconciliation.

If the provider returns only `b_mu` and `g_mumu`, with no tangent handle or
`g_Amu`, the formal-manifold composition is unsupported and must fail closed.
Legacy routes may retain their legacy Phase-I model.

## Required Phase-II model

When Phase I already has complete geometry, Phase II must not query that
geometry again. Phase II remains the mandatory second-order response stage. It
consumes the ordinary coordinate-Hessian blocks already returned by the
established Phase-II oracle contract, with explicit type

\[
Q^+_{\mathcal B}
=
\begin{bmatrix}
Q_{AA} & Q_{A\mathcal B}\\
Q_{\mathcal BA} & Q_{\mathcal BB}
\end{bmatrix},
\qquad
Q_{ij}=\partial_i\partial_jE.
\]

At exact zero-coordinate growth, the pure-old ordinary-coordinate block
`Q_AA` is reusable after logical remapping. The candidate mixed and
candidate--candidate blocks are the candidate-specific Phase-II second-order
information. They must retain `ordinary_coordinate_hessian` provenance and
must not be relabeled as covariant Hessian or inverse curvature.

For a singleton or batch, Phase II solves the full second-order metric trust
problem

\[
\Delta E_{2,\mathcal B}^{\rm qc}
=-
\min_{\delta}
\left[
(b^+_{\mathcal B})^{\mathsf T}\delta
+\frac12\delta^{\mathsf T}Q^+_{\mathcal B}\delta
\right]
\]

subject to

\[
\delta^{\mathsf T}G^+_{\mathcal B}\delta\leq\rho^2.
\]

Equivalently, the regularized/trust KKT solve has the form

\[
\left(Q^+_{\mathcal B}+\lambda G^+_{\mathcal B}\right)\delta
=-b^+_{\mathcal B},
\qquad
\lambda\geq0,
\]

with `lambda` chosen by the declared trust policy. Eliminating the old active
coordinates gives the corresponding second-order Schur response of
`Q + lambda G`. Preserve the direct full solve and Schur residual diagnostics;
they must agree within tolerance.

The metric-only response remains diagnostic:

\[
R_{\mathcal B}^{(1)}
=\widetilde b_{\mathcal B}^{\mathsf T}
(S_{\mathcal B}+\lambda_G I)^{\dagger}
\widetilde b_{\mathcal B},
\]

but it is not the authoritative Phase-II score when the established Phase-II
second-order receipt is valid.

### Mandatory optimizer curvature recycling

Separate from the Phase-II coordinate Taylor model, formal-manifold growth must
always initialize the optimizer's inverse raised-curvature model as

\[
B_0^+=\operatorname{diag}(B_A,\beta I)
\]

in the inherited plus residual orthonormal physical frame. This is mandatory,
not optional. Its mixed zero block is `unknown_prior_zero`, and the candidate
scale is a `regularized_curvature_prior`. If the previous state was reset or
has no valid transported `B_A`, construct the route's declared isotropic
`B_A=\beta_A I`; the algorithm still has one deterministic curvature model.

Do not combine `Q^+` and `B_0^+` as though one were the inverse of the other.
`Q^+` is the Phase-II ordinary-coordinate second-order selector model;
`B_0^+` is the optimizer's physical-frame inverse-curvature prior, subsequently
updated by transported BFGS secants.

The fixed phase roles are:

```text
Phase I: query-closed first-order differential plus full Gram response
Phase II: mandatory ordinary-Hessian second-order Schur/trust response
Optimizer growth/refit: mandatory recycled physical-frame B prior plus BFGS
```

The new route may reuse candidate second-order fields returned by the current
Phase-II query contract. It must not introduce an additional duplicate
candidate-curvature query merely because the response is consumed by a new
model.

## Required combinatorial batch/Schur model

Build one population Gram workspace before subset search. If Phase-I records
carry common-state tangent handles, compute

\[
(G_{CC})_{\mu\nu}
=\langle t_\mu,t_\nu\rangle_{\rm FS}
\]

by classical contraction and charge zero queries. Assemble the full metric and
ordinary-Hessian blocks for every feasible subset from the shared workspace,
then apply the mandatory Phase-II second-order trust solve.

The route's batch authority is combinatorial over the configured feasible
Phase-II population and cardinality cap:

\[
\mathcal B^*
\in
\arg\max_{\substack{
\mathcal B\subseteq\mathcal C_2,\\
1\leq|\mathcal B|\leq B_{\max},\\
\mathcal B\ \mathrm{feasible}
}}
\frac{\Delta E_{2,\mathcal B}^{\rm qc}}{1+K_{\mathcal B}}.
\]

Here `C_2` is the deterministic Phase-II survivor population after identity,
symmetry, and rank gates. Evaluate every feasible subset under the configured
cap, use deterministic tie breaking, and report the number of subsets searched.
Do not use greedy admission as the authoritative batch route. Greedy may remain
only as explicitly labeled legacy telemetry or an ablation outside this
composition.

Requirements:

1. reuse active, active-candidate, candidate-diagonal, pair, and gradient data;
2. preserve insertion-position and duplicate-child identity rules;
3. retain one subset-solve cache;
4. use combinatorial subset argmax as the authoritative selection;
5. count only provider-declared missing primitive requests;
6. never infer candidate Hessian entries from Gram entries;
7. use the mandatory second-order Phase-II response when its receipt is valid;
8. fail closed when required second-order fields are unavailable; a diagnostic
   fallback may emit telemetry but cannot select or admit a batch;
9. pass selected block matrices and registry permutation directly into growth.

If a provider exposes no common tangent handle, the current candidate-pair
query path remains a legal, explicitly charged fallback.

## Accepted-growth receipt

The selected candidate or batch should carry:

```text
FormalGrowthGeometryReceipt
  state/chart fingerprints
  old/new registry mapping
  G_AA provenance
  G_AB
  G_BB
  candidate gradients
  rank rule and spectrum
  source primitive IDs
  query-free derived-field list
  missing/regularized fields
```

Validate identical state, zero new coordinates, unchanged old-gate subsequence,
insertion positions, tie map, registry, Hamiltonian where relevant, rank rule,
metric convention, and branch. On success growth must not recompute the same
Gram blocks. On failure, refresh or fail closed and charge actual replacements.

## Query-ledger requirements

Retain separate counts:

\[
N_E,\qquad
N_{\nabla E},\qquad
N_G,\qquad
N_Q,\qquad
N_{Hv},\qquad
N_{\rm cross}.
\]

Add telemetry for:

```text
unique_primitive_ids_requested
unique_primitive_ids_reused
query_free_derived_fields
primitive_to_returned_fields
primitive_to_consumer_phases
phase1_to_phase2_reuse_count
phase2_to_batch_reuse_count
batch_to_growth_reuse_count
matrix_element_diagnostics
primitive_count_reconciliation
```

Rules:

- charge each primitive once per exact identity;
- multiple derived fields do not multiply the charge;
- factorization, pseudoinversion, eigendecomposition, Schur operations, subset
  search, qBroyden, and BFGS are free;
- different states and branches are different primitives;
- cross-state tangent work is `N_cross`;
- ordinary coordinate second derivatives are `N_Q`;
- candidate HVP work is `N_Hv`;
- matrix-element counts remain diagnostic unless the provider proves a
  one-entry-per-query contract;
- route-off behavior and counts remain unchanged.

## Implementation boundaries

Prefer a narrow module such as:

```text
pipelines/static_adapt/selector_query_closure.py
```

or extend a suitable typed module without duplication. Expected integration
surfaces are:

```text
pipelines/static_adapt/formal_manifold_warm_start.py
pipelines/static_adapt/formal_manifold_exact_backend.py
pipelines/static_adapt/adapt_pipeline.py
pipelines/static_adapt/route_a_funnel.py
pipelines/static_adapt/route_a_schur_selector.py
pipelines/static_adapt/selector_measurement_proxy.py
pipelines/static_adapt/adapt_candidate_record_cache.py
pipelines/scaffold/hh_continuation_scoring.py
pipelines/scaffold/hh_continuation_types.py
```

Do not add more matrix mathematics to nested `adapt_pipeline.py` closures.
Do not overload `phase2_joint_geometry_reuse` with untyped mixed semantics;
version it with block provenance or split metric, differential, and curvature
prior records. Do not serialize tangent vectors or provider handles.

## Required tests

### Primitive-set invariance

With a counting mock oracle that returns a tangent handle, compare baseline and
enriched paths on the same candidates. Assert identical new primitive-ID sets
and counts while the enriched path produces nontrivial `G_AB`, Schur residual,
rank, conditioning, and query-closed scores.

### Phase-I to Phase-II reuse

Assert Phase II consumes the Phase-I receipt with no duplicate metric query.
Mutate each fingerprint field and verify fail-closed refresh or explicit
fallback charging.

### Population and batch closure

With three common-state tangent handles, verify every `G_CC` entry against a
direct contraction and assert zero pair queries. Repeat with a scalar-only
provider and charge only missing pair primitives.

### Growth handoff

Admit a singleton and batch. Verify growth consumes selected blocks without a
duplicate metric query. Separately mismatch insertion, state, registry, tie
map, Hamiltonian, and branch; each must invalidate reuse.

### Second-order and curvature typing

Verify both authoritative second-order objects remain distinct:

```text
Phase-II Q provenance == ordinary_coordinate_hessian
optimizer B provenance == transported_or_regularized_inverse_raised_curvature
optimizer mixed block status at growth == unknown_prior_zero
no duplicate Phase-II second-order primitive ID is charged
ordinary Hessian is never relabeled covariant Hessian or optimizer B
```

Verify the Phase-II direct trust solve and its Schur-eliminated form agree, and
verify the optimizer always receives a valid `B_0^+`, including reset/rank-zero
initialization.

### Accounting reconciliation

Verify primitive IDs reconcile with `actual_operator_probe_count`; derived
fields and matrix-element diagnostics do not automatically affect totals;
cache hits do not double-charge; different states/branches charge separately;
statevector shortcuts are labeled; route-off histories remain unchanged.

### Numerical model tests

Verify singleton and batch Schur formulas and query-closed scores against direct
constrained solves. Include rank-zero, redundant, near-threshold, permuted
insertion, tied-parameter, and empty-active-context cases.

Retain and extend at least:

```text
test/test_hh_continuation_scoring.py
test/test_static_adapt_controller_measurement_work.py
test/test_static_adapt_route_a_funnel.py
test/test_adapt_candidate_record_cache.py
test/test_static_adapt_formal_manifold_warm_start.py
test/test_static_adapt_formal_manifold_exact_backend.py
```

Run focused unit/integration tests only. Do not launch Paper-I jobs.

## Acceptance criteria

Implementation is complete only when:

1. Phase I consumes the augmented Gram for both macro and Pauli-child
   candidate-position records whenever provider capabilities allow it.
2. Phase II reuses the same receipt with no duplicate metric query.
3. Batch/Schur contracts candidate-pair Gram entries classically when handles
   are present and uses combinatorial subset argmax.
4. Accepted growth reuses selected blocks without another same-state query.
5. Enriched and baseline primitive-ID sets are identical in zero-increment
   fixtures.
6. Missing required capabilities fail closed for the formal composition;
   legacy fallback remains available only to legacy routes.
7. Phase II retains a mandatory ordinary-Hessian second-order Schur/trust
   response.
8. Optimizer growth retains mandatory inverse-curvature recycling and BFGS.
9. Ordinary Hessian, covariant Hessian, inverse-curvature prior, Gram, and
   qBroyden prediction remain separately typed.
10. No duplicate candidate second-order query is added beyond the established
    Phase-II primitive set.
11. Query counts use logical estimator primitives, not samples or unqualified
   matrix entries.
12. State, branch, insertion, registry, tie-map, and Hamiltonian mismatches
    invalidate reuse.
13. Candidate-cache serialization stays bounded and portable.
14. Existing routes/defaults remain unchanged when the composition is off.
15. Telemetry identifies Phase-I-to-Phase-II-to-batch-to-growth reuse.
16. Focused existing and new tests pass.

## Explicit non-goals

This task does not authorize manuscript edits, scientific runs, default
activation, candidate Jacobi/Riemann-curvature lookahead, duplicate candidate
Hessian/HVP queries beyond the established Phase-II frontier, relabeling
qBroyden as measured QGT, equality-of-value reuse, greedy batch authority, or
changing prune authority beyond consuming the route-scoped scores.

## Required final report from the main agent

Report:

1. exact files and typed interfaces changed;
2. authoritative Phase-I, Phase-II, and batch formulas for the experimental
   composition;
3. before/after primitive-ID sets and counts on focused fixtures;
4. exact, reused, predicted, prior, and missing fields;
5. every fallback that still requires a new oracle query;
6. test commands/results;
7. unresolved hardware-provider capability gaps.

Do not call the route globally query-optimal from one fixture. The supported
claim is that it exhausts the provider-declared query-free closure, with tests
proving no increase in unique estimator-oracle primitives whenever the stated
closure capabilities hold.
