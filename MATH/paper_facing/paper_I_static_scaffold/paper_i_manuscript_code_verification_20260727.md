# Paper-I manuscript-to-code verification process

## Objective

Verify that the algorithm described in
`MATH/paper_details/Paper_I.tex` matches both:

1. the current RA-ADAPT implementation; and
2. the implementation and telemetry that produced each reported result.

The second comparison is essential. Current code may have been corrected after
the reported trajectories were generated.

## Scope and authority

- Work read-only. Do not edit the manuscript, code, tests, manifests, or result
  files.
- Do not launch scientific runs unless the user separately authorizes a
  diagnostic.
- Use the active local checkout:
  `/Users/jakestrobel/local_repos/Holstein_test_fullclone_3`.
- Start from the repository router and the nearest applicable `AGENTS.md`.
- Treat `MATH/paper_details/Paper_I.tex` as the manuscript authority.
- Treat code as evidence of current behavior, not as authority for
  reader-facing terminology.
- Treat executed result records and candidate telemetry as authority for
  historical behavior. A configuration declaration alone does not prove that
  the declared operation affected ranking or selection.
- Preserve the distinction between the
  **undecomposed-generator representation** and the
  **single-Pauli-word representation**.
- Do not infer equal state-tangent norms from operator normalization.
- Do not make publication or artifact-promotion judgments.

## Verification method

For each work package:

1. Quote the exact manuscript claim or equation being tested.
2. Locate the current implementation path from public entry point to the
   operation that controls the behavior.
3. Identify tests that exercise the same path.
4. When the claim also describes reported results, trace the displayed result
   to its source record and inspect executed telemetry separately.
5. Construct one minimal mathematical or numerical witness when static reading
   alone does not settle the claim.
6. Classify the result as:
   - `matches current implementation`;
   - `matches reported execution`;
   - `mismatch`;
   - `not recoverable from retained evidence`; or
   - `unclear`.
7. Recommend the smallest correction. Separate manuscript corrections from
   implementation corrections.

Do not broaden one discrepancy into a repository-wide audit. Complete each
package independently so findings can be acted on without waiting for every
other package.

## Work packages

### 1. Generator normalization and candidate representation

Verify:

- the normalization convention for undecomposed operator-pool generators;
- whether normalization occurs before or after encoding, symmetry filtering,
  padding checks, and duplicate removal;
- whether a single-Pauli-word candidate is \(c_\mu P_\mu\), a normalized
  version of \(c_\mu P_\mu\), or \(P_\mu\) with its coefficient absorbed into
  the variational parameter;
- whether RA-ADAPT and its append-only and Geo-ADAPT comparators receive
  identical candidate generators under each matched representation;
- whether signs, scalar coefficients, or duplicate Pauli words can change the
  candidate identity or score; and
- whether any code or test incorrectly treats normalized generators as proof
  that the state-dependent diagonal Gram entries \(G_{rr}\) are equal.

Report the exact implemented normalization formula and the stage at which it is
applied.

### 2. Phase-III retained support and accepted-refit whitening

Verify:

- which Fubini--Study Gram matrix defines the Phase-III retained tangent
  support;
- the relative eigenvalue threshold and the matrix against which the largest
  eigenvalue is measured;
- whether the supported pseudoinverse is restricted to the retained range;
- which Gram matrix is diagonalized after proposal acceptance;
- whether the accepted refit uses
  \(W=V\Lambda^{-1/2}\) and
  \(\theta(y)=\theta^{(0)}+Wy\);
- whether the whitened chart is fixed for the entire refit and discarded
  afterward; and
- whether any excluded directions are later reintroduced by a full-coordinate
  cleanup step.

Keep the Phase-III response solve distinct from the post-acceptance optimizer
coordinate system.

### 3. Phase-III trust-region numerical solve

Verify:

- the joint candidate--active quadratic model;
- the Schur reduction and supported pseudoinverse;
- the nonnegative KKT multiplier convention, including the
  \(\lambda=0\) interior case;
- complementarity and feasibility handling;
- any curvature shift, damping, regularization, or negative-curvature branch;
- the root-solving interval, tolerances, and failure behavior;
- trust-radius bounds and expansion safeguards; and
- whether the implemented radius update matches the displayed ratio after all
  guards are applied.

Report mathematical behavior, not only function names or configuration labels.

### 4. Active-gradient acquisition and logical-estimator accounting

Verify:

- whether active-ansatz gradients are measured in every Phase-III outer
  iteration, reused from another calculation, or set to zero by stationarity;
- whether rejected evaluated candidates are counted;
- whether repeated optimizer callbacks and repeated Hamiltonian evaluations are
  counted;
- which initialization, endpoint, terminal-selector, and diagnostic evaluations
  are excluded; and
- whether the equations in Appendix `app:estimator_queries` reproduce the
  executed accounting for each displayed representation.

Separate logical estimator queries from Pauli terms, grouped measurement
settings, shots, and compiled circuit resources.

### 5. Phase-wise cost application and measurement reuse

Verify current and historical behavior separately:

- the score used to sort Phase I;
- the score used to sort Phases II and III;
- whether the cost factor is a multiplier or a denominator in each executed
  implementation;
- whether the cost baseline is fixed independently of candidate evaluation
  order;
- which observables are reused across phases when the representation is
  unchanged; and
- which parent measurements are invalidated when retained parents are split
  into single-Pauli-word children.

For historical results, rely on candidate telemetry when available. Record
explicitly when rejected Phase-I candidates were not retained and exact
counterfactual shortlist membership cannot be reconstructed.

### 6. Hamiltonian-proportional generator handling

Verify:

- whether a generator proportional to the full Hamiltonian can enter each
  relevant operator pool;
- whether it can be selected as a terminal append, a nonterminal append, or an
  insertion;
- how the implementation handles its zero commutator with \(H\);
- whether duplicate or zero-tangent guards remove it; and
- whether any manuscript statement about this special case is broader than the
  implemented behavior.

### 7. Insertion protocol

Perform this package only after the final insertion configuration and results
are identified.

Verify:

- the stagnation trigger and patience rule;
- eligible insertion positions;
- commutation reduction;
- whether append remains an eligible position;
- tie breaking;
- the relation between the insertion proposal, accepted refit, and subsequent
  return to append-mode selection; and
- whether the displayed `always insertion` and `plateau insertion` labels
  correspond to the executed policies.

### 8. Compiled-resource convention

Verify:

- coupling graph or backend topology;
- initial layout and routing;
- basis gates;
- transpiler optimization level and seed;
- reference-state preparation;
- the precise definition of a two-qubit gate for \(N_{2q}\);
- the algorithms used for \(D_{2q}\) and total circuit depth \(D_c\); and
- whether all compared methods use the same compilation protocol.

### 9. Operator-pool inventory and lane attribution

Verify:

- the number of undecomposed parent generators;
- the number of generated Pauli terms and symmetry-valid
  single-Pauli-word candidates;
- duplicate removal and multi-parent child attribution;
- physical-lane membership for parents and children;
- how a child generated by several parents is assigned for lane protection and
  operator-family summaries; and
- whether pool-composition figures and heat maps use the same attribution rule
  as selection.

## Required output

Create one report section per completed work package:

```text
## Verification result: <work-package name>

Status:
Manuscript claim:
Current implementation evidence:
Reported-execution evidence:
Mathematical or numerical witness:
Agreement:
Smallest manuscript correction:
Smallest implementation correction:
Unrecoverable evidence:
Open questions:
```

Every evidence statement must include an exact file path and line number or an
exact result-record path plus the relevant field. Label inference explicitly.
If no correction is required, write `None`; do not manufacture cleanup work.

## Definition of done

A work package is complete when:

- the manuscript claim has been quoted exactly;
- current behavior and historical executed behavior have been separated;
- the complete controlling path has been traced;
- available tests and result telemetry have been checked;
- uncertainties are named without guessing; and
- the recommendation is narrow enough to apply without reopening unrelated
  manuscript structure.

## True stop conditions

Stop only when:

- scientific semantics are genuinely ambiguous and different interpretations
  would change the conclusion;
- the displayed result cannot be resolved to any source record;
- required files are irrecoverably unavailable; or
- a new diagnostic run is necessary and has not been authorized.

Ordinary missing tests, stale names, or weak documentation are findings, not
automatic stop conditions.
