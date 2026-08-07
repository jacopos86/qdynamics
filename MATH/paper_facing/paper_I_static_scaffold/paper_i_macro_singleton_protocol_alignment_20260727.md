# Paper-I macro/single-Pauli-word protocol alignment

Date: 2026-07-27  
Status: planning and provenance reconciliation only  
Scope: final Hubbard--Holstein RA-ADAPT comparisons using undecomposed
generators and single-Pauli-word generators

## Purpose

This note separates confirmed historical macro/single-Pauli-word differences
from differences that were incorrectly inferred. Its purpose is to define the
minimum alignment needed if the final Paper-I comparison is intended to isolate
candidate representation.

The underlying read-only audit is
`paper_i_manuscript_code_verification_20260727.md` together with the returned
audit report supplied to the manuscript-editing task. This note does not
authorize runs, result replacement, or manuscript edits.

## Summary

Four possible comparison confounders require attention:

1. trust-radius calibration;
2. the candidate-coordinate chart used for Phase-III prediction;
3. the numerical representation of the Phase-III supported solve;
4. the Phase-I parent population and downstream candidate supply.

The displayed macro and single-Pauli-word results already use the same
historical phase-wise cost scope, estimator-accounting convention, and Qiskit
compilation identity. Those items are not macro-versus-single-Pauli-word
confounders.

## 1. Trust-radius calibration

### Confirmed historical difference

- Displayed macro trajectory: the realized displacement was calibrated from an
  endpoint Fubini--Study overlap and therefore required an overlap acquisition.
- Displayed single-Pauli-word trajectory: the realized displacement was
  evaluated in the retained source-point Gram metric without an endpoint
  overlap.

These are different trust-update policies and can produce different subsequent
radii even after identical predicted and refitted steps.

### Alignment requirement

Replacement macro and single-Pauli-word trajectories should use the same
source-point-metric, no-endpoint-overlap trust update unless the author
explicitly chooses another common policy.

## 2. Append-coordinate chart versus exact ordered-insertion chart

For a candidate generator \(A\) at logical position \(p\), the exact local
family is

\[
U_{k,r}(\alpha,\delta\boldsymbol\theta)
=
U_{k,>p}(\boldsymbol\theta^\star_{>p}
        +\delta\boldsymbol\theta_{>p})
e^{-i\alpha A}
U_{k,\le p}(\boldsymbol\theta^\star_{\le p}
           +\delta\boldsymbol\theta_{\le p}).
\]

The candidate tangent and Hessian depend on \(p\) because the gates after the
candidate dress its state-space direction. At the appended position,
\(U_{k,>p}=I\), so the exact ordered-insertion chart reduces to the append
chart.

### Confirmed historical difference

- Displayed macro receipt:
  `append_candidate_after_current_ansatz_v1`.
- Displayed single-Pauli-word receipt:
  `exact_ordered_insertion_zero_angle_v1`.

The macro receipt proves that its response derivatives were generated in the
append-coordinate chart. The single-Pauli-word receipt proves that its
derivatives were generated at the recorded position in the exact ordered
product.

### What this does and does not yet prove

The receipt alone does not establish whether the historical macro procedure
enumerated only the appended position or enumerated interior positions and
then scored them with append-chart derivatives.

If it enumerated interior positions, the historical procedure effectively did
the following:

1. construct candidate-position records;
2. predict their response using the appended-candidate chart;
3. admit the winner at its recorded interior position.

In that case, the predicted Hessian and active-coordinate response would not
belong to the circuit actually refitted after insertion. If the macro procedure
considered only the appended position, the append chart was exact and there is
no chart error for that trajectory.

### Required provenance check

Resolve from the macro candidate telemetry whether any accepted or scored
record had \(p<n_k\). The combination

- interior \(p\), and
- `append_candidate_after_current_ansatz_v1`

is the material mismatch.

### Alignment requirement

All replacement insertion trajectories should use
`exact_ordered_insertion_zero_angle_v1` at every recorded position. Append-only
records then arise as the \(p=n_k\) special case of the same implementation
path.

## 3. Phase-III selector solve versus accepted-refit whitening

Two separate coordinate transformations must not be conflated.

### Phase-III selector

The Phase-III selector predicts the candidate-plus-active response inside the
trust region.

- Displayed macro selector:
  `supported_metric_whitened_eigh_v1`, with a \(10^{-9}\) metric ridge.
- Displayed single-Pauli-word selector:
  projected generalized solve on the retained raw-Gram support, without
  selector whitening.

Whitening the retained metric support is algebraically a coordinate change. If
the support, eigenvalue threshold, ridge, quadratic model, and numerical
tolerances are identical, a whitened eigensolve and an unwhitened generalized
solve should represent the same physical trust-region step.

The historical implementations are not guaranteed identical because the macro
solver added a metric ridge and used a different numerical path. This is a
smaller confound than the trust-update or insertion-chart differences, but it
should be removed from a representation-only comparison.

### Accepted-refit optimizer

Accepted-refit whitening is a separate operation performed after admission.
Powell optimizes a fixed, full-enlarged-ansatz whitened chart. The audit did not
identify a macro-versus-single-Pauli-word difference in this accepted-refit
coordinate system.

Therefore, “macro selector whitening” does not imply that only the macro inner
optimizer was whitened. The confirmed selector difference occurs before
admission. The accepted-refit chart should be verified independently from the
result receipts, but it is not presently a confirmed mismatch.

### Alignment requirement

Replacement trajectories should use:

- the same retained-support rule;
- the same projected generalized Phase-III solve;
- the same stabilization convention;
- the same complete accepted-ansatz whitened Powell refit.

The Phase-III receipt must distinguish the minimum generalized-curvature
stabilization shift \(\kappa\) from the additional trust-boundary multiplier
\(\lambda\). It may also record their sum, but a single combined
`trust_lambda` field is insufficient to determine whether the trust boundary
was active.

## 4. Parent population and comparator candidate supply

Three comparisons must be distinguished.

### 4.1 Macro RA-ADAPT versus macro Append-ADAPT

The audit reports:

- macro RA-ADAPT Phase-I population: 102 parents at cutoff 3 and 148 parents at
  cutoff 7;
- macro Append-ADAPT population: 123 and 171 unfiltered parents.

Thus the historical macro comparison did not expose both methods to the same
undecomposed parent population. The audit does not yet identify which 21 or 23
parents were removed or whether the filter was an intentional method policy or
a stale historical configuration.

This is a direct comparator mismatch if the manuscript presents the macro
comparison as using an identical operator pool.

### 4.2 Single-Pauli-word RA-ADAPT versus single-Pauli-word Append-ADAPT

Single-Pauli-word RA-ADAPT uses a staged supply:

1. Phase I scores the 123/171 undecomposed parents.
2. It retains eight parents.
3. Only those eight parents are split and symmetry guarded.
4. The resulting dynamic child population enters Phase II.
5. Four children proceed to Phase III.

At round zero, the audit found 37 Phase-II records at cutoff 3 and 56 at cutoff
7, including already-single-Pauli-word parents.

Single-Pauli-word Append-ADAPT instead scans the global single-Pauli-word child
pool. Consequently, both methods ultimately append single-Pauli-word
generators, but they do not evaluate the same candidate population:

- RA-ADAPT can select only children of its eight retained parents;
- Append-ADAPT can select any child in the global child pool.

This difference is partly the intended staged-restriction method, not merely a
historical accident. The paper should call this a matched candidate
representation, not an identical evaluated candidate population.

### 4.3 Macro RA-ADAPT versus single-Pauli-word RA-ADAPT

If macro RA-ADAPT begins from 102/148 parents while single-Pauli-word RA-ADAPT
begins from 123/171 parents, their difference is not solely candidate
representation. It also includes parent-pool exposure.

For a representation-focused comparison, both RA-ADAPT variants should begin
from the same 123/171 parent pool:

- macro condition: advance retained parents without Pauli splitting;
- single-Pauli-word condition: retain parents, split them, and advance the
  symmetry-valid children.

The downstream candidate populations will still differ by design, but the
upstream parent supply will be controlled.

### Required provenance check

Identify the excluded 21/23 macro parents and the configuration field that
removed them. Decide whether that filtering belongs to the intended macro
representation or is a historical artifact.

## Items already aligned

### Phase-wise cost scope in the displayed results

Both displayed historical families used

\[
S^{(1)}=\Delta E_1,\qquad
S^{(2)}=\Delta E_2/K_2,\qquad
S^{(3)}=\Delta E_3/K_3.
\]

Thus Phase I was unweighted and Phases II/III were resource weighted in both
macro and single-Pauli-word results. This is not a historical
macro-versus-single-Pauli-word difference.

The current implementation applies the cost factor in all three phases. The
author must choose one common convention for replacement runs, but that is a
historical-versus-current-method decision.

### Estimator accounting

The logical-estimator accounting convention is common across the displayed
families. Different values of \(S\) are expected because the candidate
populations and acquired derivatives differ. No accounting change is required
merely to align macro and single-Pauli-word runs.

### Compiled resources

The displayed rows share the same compile identity:

- optimization level 0;
- transpiler seed 7;
- common basis-gate set;
- reference-state preparation included;
- no coupling map, initial layout, or routing constraint.

The absence of a coupling map should be stated in the manuscript, but it is not
a macro-versus-single-Pauli-word mismatch and does not by itself require a
rerun.

### Hessian and active gradient

Both displayed families used exact ordered-product derivative propagation and
the residual active-coordinate gradient. The corresponding defect is confined
to the manuscript equations.

For replacement runs, the active-gradient policy remains an author decision.
Setting the residual active block to zero defines a stationary-source
approximation distinct from the displayed implementation. If selected, both
representations must omit the active-gradient acquisition and use the
coupling-only Schur response; the estimator accounting must then remove those
active-gradient occurrences. The zero-residual and measured-residual policies
must not be mixed across the macro and single-Pauli-word comparison.

### Single-Pauli-word representative

For every single-Pauli-word candidate, the nonzero parent coefficient is
absorbed into the variational coordinate and the executed child is the
canonical unit Pauli representative. Replacement RA-ADAPT and Append-ADAPT
trajectories must use this same convention. It removes coefficient-induced
rescaling from conventional ADAPT's raw-gradient ranking; the RA-ADAPT energy
models are invariant under the corresponding one-coordinate
reparameterization.

## Replacement-run alignment checklist

Before interpreting macro/single-Pauli-word differences as representation
effects, verify that both final result families use:

1. the same unfiltered Phase-I parent pool;
2. exact ordered-position derivatives for every candidate-position record;
3. the same retained-support threshold and projected generalized Phase-III
   solve;
4. the same curvature-stabilization convention, with \(\kappa\) and the
   trust-boundary multiplier \(\lambda\) recorded separately;
5. the same source-metric, no-endpoint-overlap trust update;
6. the same phase-wise cost scope chosen by the author;
7. the same complete accepted-ansatz whitened Powell refit and budget;
8. the same stopping rule and execution horizon;
9. the same canonical unit-Pauli representative for every single-Pauli-word
   candidate in RA-ADAPT and Append-ADAPT;
10. the same measured-residual or stationary-source active-gradient policy;
11. the already common estimator-accounting and Qiskit compilation
    conventions.

Every replacement result should record the derivative chart, trust-update
policy, Phase-III solver identity, parent-pool identity, cost scope, and
accepted-refit coordinate system explicitly, together with separate
curvature-stabilization and trust-boundary multipliers.

## Unresolved questions

1. Did the historical macro plateau-insertion trajectory score any interior
   position using append-chart derivatives?
2. Which 21/23 parents were excluded from the macro RA-ADAPT population, and
   why?
3. Do the displayed macro and single-Pauli-word receipts confirm the same
   complete accepted-refit whitening policy?
4. Should the final method leave Phase I unweighted or apply the resource factor
   in all three phases?
5. Should replacement Phase III use the measured residual active gradient or
   impose the stationary-source approximation \(\mathbf g_\theta=\mathbf0\)?

Files to edit: None. This note records questions and alignment requirements; it
does not authorize changes to code, runs, results, or the manuscript.
