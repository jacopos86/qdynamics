# Integrated Geometric Recycling Implementation Contract

**Date:** 2026-07-11  
**Status:** implementation-ready design contract with formally resolved warm
start; no code changes made  
**Behavioral scope:** one atomic experimental stack containing a labeled
qBroyden pullback-metric predictor, Riemannian objective-Hessian recycling, and
a transported quotient-manifold warm start

## Formal Authority

The complete derivation, provenance ledger, typed pseudocode, and 26
falsification tests are in
`paper_i_formal_manifold_warm_start_20260711.md`. If this implementation
contract and that formal note disagree, the formal note governs.

The formal result corrects three earlier assumptions:

1. published qBroyden is a gradient-outer-product low-pass recurrence, not a
   QGT secant update;
2. a block-diagonal enlarged inverse Hessian is a regularized prior, not an
   exact inverse-block identity;
3. the optimizer must combine metric and objective curvature in a resolved
   orthonormal physical tangent frame so the metric is not applied twice.

## Objective

Implement one complete experimental optimizer stack that carries geometric and
energy-curvature information across both inner optimization steps and outer
adaptive ansatz growth.

The stack must include all of the following before it is exposed as runnable:

1. exact Gram-block extension at zero-initialized ansatz growth;
2. exact tangent-projector rank update at that same point;
3. qBroyden prediction of the coordinate pullback between genuine geometric
   anchors, with objective-dependent provenance;
4. Riemannian objective-Hessian recycling in a transported orthonormal physical
   tangent frame;
5. a manifold warm start for the metric, tangent representation, and energy
   Hessian;
6. sparse refresh and full-refresh safeguards;
7. expanded query accounting;
8. focused unit, integration, and compatibility tests;
9. an isolated typed composition profile that leaves existing routes
   unchanged.

This is an all-at-once feature contract. Internal modules may be developed and
tested separately, but no partial route or partial public configuration should
be activated before the complete stack passes its acceptance gate.

## Difficulty Assessment

| Component | Difficulty for an AI coding agent | Reason |
|---|---:|---|
| qBroyden pullback predictor | Moderate | Published gradient-outer-product update; objective dependent and valid only between declared anchors |
| Adaptive inverse-Hessian recycling | Moderate | Published adaptive-VQA precedent; requires dimension remapping and stable secant guards |
| Exact Gram block extension | Low to moderate | Exact identity at zero insertion; current geometry objects already exist |
| Exact tangent-projector extension | Moderate | Straightforward linear algebra once a stable tangent representation exists |
| Riemannian Hessian integration | Hard | Metric, gradient, Hessian, and transport must use one tensor convention |
| Manifold transport in exact-state simulation | Moderate to hard | Tangent frames and cross-state overlaps are available but need extraction and gauge handling |
| Measurement-faithful transport | Research-hard | Cross-state geometric observations, uncertainty, and refresh accounting need explicit primitives |
| Complete integrated stack | Hard but tractable | Feasible with this contract and narrow tests; not a routine optimizer addition |

The implementation risk is primarily semantic, not code volume. An agent can
write the numerical routines; it must not guess metric normalization, tangent
transport, rank handling, or how two inverse operators combine.

## Mathematical Types

At state

\[
|\psi(\boldsymbol\theta)\rangle,
\qquad
\boldsymbol\theta\in\mathbb R^d,
\]

define horizontal tangents

\[
|t_i\rangle
=
\left(I-|\psi\rangle\langle\psi|\right)
\partial_i|\psi\rangle.
\]

The internal metric convention is the real Fubini--Study Gram matrix

\[
G_{ij}
=
\Re\langle t_i|t_j\rangle.
\]

Do not mix this silently with the convention in which the quantum Fisher
information matrix is (4G). Convert once at a public boundary and record the
normalization.

The coordinate energy gradient is

\[
b_i
=
\partial_iE.
\]

Resolve the constant-rank quotient through

\[
T_{\rho}
=
\mathcal E L,
\qquad
\mathcal E^{*_{\mathbb R}}\mathcal E=I_{\rho},
\qquad
G_{\rho}=L^{\mathsf T}L,
\]

where \(\mathcal E:\mathbb R^{\rho}\to V_x\) is an orthonormal physical
tangent frame and \(L:\mathbb R^d\to\mathbb R^{\rho}\) maps circuit-coordinate
displacements into that frame. The frame Riemannian gradient is

\[
\bar r
=
L^{+\mathsf T}b
\in
\mathbb R^{\rho}.
\]

Keep three objective-curvature objects distinct:

\[
Q_{ij}
=
\partial_i\partial_jE,
\qquad
h_{ij}
=
\partial_i\partial_jE
-
\Gamma^a{}_{ij}\partial_aE,
\qquad
\mathcal A
=
\sharp_g\circ h^{\flat}.
\]

Here \(Q\) is an ordinary coordinate Hessian, \(h\) is the covariant Hessian
bilinear form, and \(\mathcal A:V_x\to V_x\) is the raised self-adjoint
Hessian operator. They coincide only under additional conditions.

The primary positive-curvature branch stores

\[
B
\approx
\mathcal A^{-1}
\]

in the orthonormal \(\mathcal E\)-frame. Its type-correct step is

\[
\boxed{
z=-B L^{+\mathsf T}b,
\qquad
p=L^+z.
}
\]

The implementation therefore recycles:

- a predicted coordinate pullback or inverse-metric cache derived from \(L\);
- an inverse raised-objective-Hessian operator \(B\) in the physical frame, or
  one explicitly tagged direct SR1 operator \(A\);
- the orthonormal physical frame \(\mathcal E\) and coordinate map \(L\);
- rank, fingerprint, and uncertainty metadata.

Do not apply another metric inverse after \(L^{+\mathsf T}b\). In a nonsingular
active coordinate chart the equivalent expression is \(-B_{\rm coord}M b\),
not \(-B_{\rm coord}M^2b\).

## Atomic Optimizer State

Introduce a dedicated typed state rather than extending the existing diagonal
SPSA memory payload.

Conceptually:

```text
GeometricRecycleState
  schema_version
  state_handle
  energy
  theta
  logical_coordinate_registry
  state_fingerprint
  scaffold_fingerprint
  physical_frame_handle
  coordinate_to_frame_map
  coordinate_differential
  metric_normalization
  metric_rank
  metric_eigenvalues
  metric_anchor_provenance
  qbroyden_reduced_inverse
  qbroyden_anchor_age
  curvature_mode
  inverse_raised_hessian
  direct_raised_hessian
  trust_radius
  previous_frame_gradient
  previous_accepted_step
  refresh_counters
  query_counters
  invalidation_reason
  history_tail
  rollback_checkpoint
```

Derive \(G=L^{\mathsf T}L\), \(G^+=L^+L^{+\mathsf T}\),
\(\Pi=\mathcal E\mathcal E^{*_{\mathbb R}}\), and
\(\bar r=L^{+\mathsf T}b\) from the authoritative state. Do not persist
independent copies that can silently disagree.

Dense matrices must not be inserted into `Phase2OptimizerMemoryAdapter`, which
currently remaps only diagonal vectors. Create a dedicated adapter capable of:

- insertion at an arbitrary logical position;
- block insertion for a batch;
- coordinate removal after prune;
- active-window selection and merge;
- branch-local cloning;
- serialization with explicit size limits;
- fingerprint validation;
- fail-closed invalidation.

## Exact Outer-Iteration Warm Start

When a new coordinate (alpha) is introduced with

\[
|\psi_{k+1}(\boldsymbol\theta,0)\rangle
=
|\psi_k(\boldsymbol\theta)\rangle,
\]

the old projective tangent block is unchanged at that point after logical
coordinate remapping and gauge identification. Define the new horizontal
tangent by

\[
|d_A\rangle
=
\left(I-|\psi_k\rangle\langle\psi_k|\right)
\left.\partial_{\alpha}|\psi_{k+1}(\boldsymbol\theta,\alpha)\rangle
\right|_{\alpha=0}.
\]

define

\[
(q_A)_i
=
\Re\langle t_i|d_A\rangle,
\qquad
F_A
=
\Re\langle d_A|d_A\rangle.
\]

Then

\[
\boxed{
G_{k+1}(\boldsymbol\theta,0)
=
\begin{bmatrix}
G_k(\boldsymbol\theta)&q_A\\
q_A^{\top}&F_A
\end{bmatrix}
}.
\]

For a batch, replace the scalar and vector by candidate blocks:

\[
G_{k+1}
=
\begin{bmatrix}
G_k&G_{AB}\\
G_{BA}&G_{BB}
\end{bmatrix}.
\]

The true old--old block is an exact identity when the zero-coordinate
restriction holds on a neighborhood. Reusing a predicted or noisy stored block
does not improve its provenance. Only the new mixed and new--new blocks are
newly observed.

For a batch, let \(D_B\) collect new tangent columns and define

\[
R_B=(I-\Pi_A)D_B,
\qquad
S_B=G_{BB}-G_{BA}G_{AA}^+G_{AB}.
\]

The exact block projector update is

\[
\Pi_{A\oplus B}
=
\Pi_A+R_BS_B^+R_B^{*_{\mathbb R}}.
\]

Its singleton specialization uses residual tangent

\[
|r_A\rangle
=
(I-\Pi_A)|d_A\rangle,
\]

with

\[
\sigma_A
=
F_A-q_A^{\top}G_k^+q_A,
\]

the tangent projector update is

\[
\Pi_{k+1}
=
\Pi_A
+
\frac{|r_A\rangle\langle r_A|_{\mathbb R}}
{\sigma_A}
\]

when \(\sigma_A\) exceeds the resolved rank floor. At collapse, do not add a
spurious tangent rank.

The restriction identity also gives exact ordinary coordinate derivatives:

\[
\partial_iE_{k+1}(\boldsymbol\theta,0)=\partial_iE_k(\boldsymbol\theta),
\qquad
\partial_i\partial_jE_{k+1}(\boldsymbol\theta,0)
=
\partial_i\partial_jE_k(\boldsymbol\theta).
\]

It does not generally preserve the old--old covariant-Hessian block. For the
old manifold immersed in the enlarged manifold,

\[
h^+_{AA}
=
h^-_{AA}
-
\left\langle
(\operatorname{grad}E)^{\perp},
\operatorname{II}_{AA}
\right\rangle.
\]

Likewise, \(\operatorname{diag}(B,\beta I)\) is only a regularized enlarged
inverse-curvature prior. Mixed objective curvature remains unknown until
inferred from accepted secants or observed through an explicit curvature
primitive.

## Metric Recycling During Reoptimization

Initialize the pullback model from an exact enlarged Gram anchor, a configured
block approximation, or a regularized identity fallback. Record the source.

Published qBroyden uses the objective-gradient outer-product recurrence

\[
\mathsf F_{j+1}
=
(1-\epsilon_j)\mathsf F_j
+
\epsilon_j b_jb_j^{\mathsf T},
\]

not a displacement/gradient-difference secant. With \(\mathsf F=4G\), the
direct \(G\)-convention recurrence is

\[
\widehat G_{j+1}
=
(1-\epsilon_j)\widehat G_j
+
\frac{\epsilon_j}{4}b_jb_j^{\mathsf T}.
\]

Run the corresponding Sherman--Morrison inverse recurrence only on one fixed,
positive-definite resolved quotient. The implementation must preserve:

- symmetry;
- resolved positive-semidefinite behavior;
- a metric eigenvalue floor;
- bounded condition number;
- deterministic reset behavior;
- exact reporting of every refresh.

This recurrence is Hamiltonian- and gradient-scale dependent. At \(b_j=0\) it
shrinks the stored matrix even when the true QGT is constant. It is therefore
an objective-dependent preconditioner predictor, not an exact or intrinsic law
for Fubini--Study geometry. The state must distinguish:

```text
exact_measured
exact_block_extended
qbroyden_predicted
sparsely_corrected
regularized_fallback
invalid
```

qBroyden predicts the coordinate stretch \(L^{\mathsf T}L\); it never changes
the identity metric of the orthonormal physical frame. Physical-frame rotation
must come from transport or a new tangent observation. Do not add qBang
momentum in this integrated contract.

## Riemannian Hessian Recycling

Represent objective curvature in the transported orthonormal physical tangent
frame. The positive-curvature branch stores
\(B_j\approx\mathcal A_j^{-1}\); the indefinite branch stores one direct
self-adjoint SR1 operator \(A_j\). Never update both as independently
authoritative objects.

For an isometric frame transport \(Q_j\), accepted frame displacement \(s_j\),
and frame gradients \(\bar r_j=L_j^{+\mathsf T}b_j\), define

\[
y_j
=
\bar r_{j+1}
-
Q_j\bar r_j,
\qquad
\widetilde B_j
=
Q_jB_jQ_j^{\mathsf T}.
\]

Use guarded inverse Riemannian BFGS satisfying

\[
B_{j+1}y_j=s_j.
\]

The direct SR1 branch instead satisfies \(A_{j+1}s_j=y_j\) and must use an
indefinite trust-region solver.

Required guards:

- reject nonfinite secant pairs;
- reject or damp pairs with insufficient metric curvature;
- preserve metric self-adjointness;
- apply Powell damping or an equivalent positive-curvature safeguard when a
  positive-definite inverse is required;
- allow an explicitly typed indefinite SR1 diagnostic mode;
- reset after transport failure, rank change, or repeated rejected secants;
- record direct secant residuals.

At outer ansatz growth, embed the prior only as a tagged initialization. No
principal block of the enlarged inverse is generally exact. Initialize new
curvature using an explicit policy:

```text
measured_block
scaled_identity
diagonal_curvature
unavailable_reset
```

Never label a guessed Hessian block as measured.

## Manifold Transport Between Inner Steps

The manifold warm start must move the metric and Hessian models into a common
tangent frame before updating them.

### Exact-state transport implementation

For the noiseless compiled-state backend:

1. construct gauge-invariant density-tangent frames, or phase-aligned
   horizontal statevector frames, at old and new parameters;
2. orthonormalize both frames under one rank rule;
3. form the real cross-frame overlap matrix;
4. resolve active ranks using one shared tolerance;
5. obtain an orthogonal or partial-isometry alignment through polar/Procrustes
   decomposition;
6. transport vectors and operators by that alignment;
7. record principal angles and transport residuals.

For endpoint frames \(\mathcal E_j,\mathcal E_{j+1}\), let

\[
C=\mathcal E_{j+1}^{*_{\mathbb R}}\mathcal E_j
=
U\Sigma V^{\mathsf T},
\qquad
Q=UV^{\mathsf T}.
\]

Transport vectors and raised operators with \(Q\) and
\(QBQ^{\mathsf T}\). This endpoint Procrustes map is isometric and agrees with
Levi--Civita transport only to first order under the stated smooth,
constant-rank, short-step assumptions. Telemetry must not call it exact
parallel transport. Same-state metrics alone do not determine \(Q\).

### Measurement-faithful interface

Define an observation interface for:

- selected metric quadratic forms;
- selected old--new tangent overlaps;
- gradient vectors;
- optional Hessian-vector products;
- uncertainty for every measured scalar.

The exact-state backend may fill this interface analytically. A quantum backend
must charge each primitive and may return partial observations. Do not make
classical statevector access part of the production contract.

### Structural transitions

Transport is valid only within one resolved fixed-rank regime. Trigger
invalidation or broad refresh on:

- rank change;
- spectral-gap closure;
- accepted prune;
- rollback changing the scaffold;
- parameterization-mode change;
- insertion/removal remap failure;
- state or scaffold fingerprint mismatch.

## Sparse Correction and Refresh

The integrated stack must support selected exact anchors. Correct a predicted
metric through a constrained PSD update, and correct the Hessian through
measured secants or Hessian-vector products.

Refresh policy must include:

- maximum steps since exact metric anchor;
- normalized metric innovation threshold;
- principal-angle threshold;
- minimum retained eigenvalue;
- maximum condition number;
- secant residual threshold;
- optimizer model-agreement ratio;
- consecutive line-search or trust-region rejection count;
- mandatory refresh after structural transition.

The policy is fail-closed for optimizer state but fail-open for the scientific
run: invalid recycled state resets to an exact or conservative optimizer state
rather than terminating the ADAPT run.

## Integrated Inner Iteration

For each inner optimization iteration:

1. Evaluate energy and coordinate differential \(b_j\).
2. Form \(\bar r_j=L_j^{+\mathsf T}b_j\).
3. Compute \(z_j=-B_j\bar r_j\) and lift \(p_j=L_j^+z_j\), or solve the
   direct-SR1 trust-region model.
4. Globalize and retain the displacement actually accepted, not the rejected
   trial displacement.
5. Evaluate the accepted endpoint and obtain an endpoint frame or declared
   transport action.
6. Apply qBroyden only to the fixed-quotient coordinate pullback prediction,
   then overwrite or correct it when an anchor is observed.
7. Recompute \(\bar r_{j+1}\) from the endpoint differential and corrected
   \(L_{j+1}\).
8. Form transported \(s_j,y_j\) from the accepted physical displacement.
9. Apply guarded Riemannian BFGS or SR1.
10. Run metric innovation, principal-angle, spectral-gap, rank, secant, and
    model-agreement gates.
11. Emit provenance-aware telemetry and query counts.

Do not apply an Adam momentum update in addition to the recycled Hessian step in
this first contract.

## Current-Code Hook Map

### New isolated package

Conceptual ownership:

```text
pipelines/static_adapt/geometric_recycling/
  config.py
  state.py
  quotient_frame.py
  pullback_predictor.py
  objective_curvature.py
  tangent_transport.py
  refresh_policy.py
  optimizer.py
  query_accounting.py
```

Names may be adjusted to local conventions, but ownership must remain outside
`adapt_pipeline.py`.

### Existing files

`pipelines/static_adapt/optimizer_routes.py`

- add one optimizer key for the complete integrated stack;
- extend dispatch with objective, gradient, metric-observation, transport, and
  memory interfaces;
- keep existing stochastic and deterministic paths unchanged.

`pipelines/static_adapt/engine_support.py`

- register the new optimizer key only;
- do not place geometric mathematics here.

`pipelines/static_adapt/paper_i_runner.py`

- add one typed geometric-recycling configuration object;
- add one experimental composition profile;
- do not change canonical or displayed-results defaults.

`pipelines/scaffold/hh_continuation_types.py`

- do not expand `Phase2OptimizerMemoryAdapter` into dense matrix ownership;
- add a separate state/remapping protocol if a shared contract is needed.

`pipelines/scaffold/hh_continuation_scoring.py`

- expose existing tangent/Gram derivative work through a narrow reusable
  workspace interface;
- do not duplicate Phase-II score formulas;
- keep candidate mixed energy-Hessian measurements outside Phase II.

`pipelines/static_adapt/adapt_pipeline.py`

- construct objective/gradient/geometry providers;
- pass typed state into optimizer dispatch;
- remap state after admission, batching, prune, rollback, and resume;
- avoid new optimizer mathematics and closure proliferation.

`pipelines/static_adapt/cli_config.py`

- expose one experimental profile or one compact JSON config;
- do not add every numerical safeguard as a first-class top-level CLI flag.

## Query Accounting

Record unique same-state primitives separately:

\[
N_E,
\qquad
N_{\nabla E},
\qquad
N_G,
\qquad
N_{Hv},
\qquad
N_{\mathrm{cross}},
\]

where:

- (N_E) counts energy/objective evaluations;
- (N_{\nabla E}) counts gradient primitives;
- (N_G) counts metric primitives;
- (N_{Hv}) counts Hessian-vector or curvature observations;
- (N_{\mathrm{cross}}) counts cross-state tangent observations.

Classical qBroyden, BFGS/SR1, transport, factorization, and matrix solves add no
quantum queries. Reused exact blocks are charged once. Different states are
different primitives. Beam accounting remains per winning branch under the
existing Paper-I convention.

Every result must report:

- exact versus predicted versus corrected matrix elements;
- refresh count and reasons;
- rejected update count;
- metric and Hessian rank/condition diagnostics;
- energy evaluations and gradients;
- expanded query work;
- whether any statevector-only shortcut was used.

## Tests

All 26 falsification tests in
`paper_i_formal_manifold_warm_start_20260711.md` are mandatory. The compact
categories below highlight integration coverage and do not replace that suite.

### Mathematical unit tests

1. Metric normalization and QFIM conversion.
2. Quotient factorization \(T_{\rho}=\mathcal E L\) and redundant-coordinate
   lift behavior.
3. Exact singleton/batch Gram extension and block projector update.
4. Covariant-Hessian second-fundamental-form counterexample.
5. Enlarged inverse-Hessian block counterexample.
6. qBroyden direct/inverse equivalence with the QFIM-to-\(G\) factor of four.
7. qBroyden zero-gradient and Hamiltonian-rescaling nongeometry tests.
8. Type-correct contraction \(z=-B L^{+\mathsf T}b\), \(p=L^+z\), including
   failure under an extra metric application.
9. Transported accepted-displacement secant construction.
10. Inverse BFGS secant satisfaction and physical-frame self-adjointness.
11. Powell damping under poor curvature.
12. Direct SR1 indefinite trust-region fixture.
13. Density-tangent Procrustes gauge, principal-angle, and frame-rotation tests.
14. Spectral-gap and rank-change invalidation.

### State remapping tests

15. Singleton insertion.
16. Batch insertion.
17. Arbitrary insertion position.
18. Prune removal.
19. Rollback restoration.
20. Branch-local clone isolation.
21. Resume serialization and fingerprint validation.

### Integrated optimizer tests

22. Quadratic objective with fixed Euclidean metric.
23. Quadratic objective in a nonorthogonal coordinate chart.
24. Small parameterized quantum-state fixture with exact (G) and (H).
25. Exact refresh versus recycled trajectory comparison.
26. Forced innovation failure and reset.
27. Full-refit and final-refit paths.
28. Query accounting changes when refresh cadence changes.

### Compatibility tests

29. Existing Powell path unchanged.
30. Existing SPSA/QNSPSA paths unchanged.
31. Existing Route-A score and selection payloads unchanged when the profile is
    off.
32. Existing checkpoint readers tolerate absence of geometric state.
33. New checkpoint readers fail closed on malformed geometric state.

## Atomic Acceptance Gate

Do not expose the profile until all conditions pass:

1. All mathematical, remapping, integration, and compatibility tests pass.
2. Existing optimizer routes show no behavioral drift when the profile is off.
3. One deterministic end-to-end fixture completes admission, full refit,
   prune/no-prune handling, checkpoint, resume, and final refit.
4. Query accounting reconciles exact anchors, predicted updates, and cache hits.
5. Statevector-only and measurement-faithful paths are labeled distinctly.
6. No qBang momentum is silently active.
7. No candidate Jacobi-curvature lookahead is silently active.
8. qBroyden output is labeled `predicted`, never `measured_qgt` or
   `parallel_transport`.
9. The step implementation applies the metric exactly once through
   \(L^{+\mathsf T}b\).
10. The new profile remains experimental and nondefault.

The complete stack may be committed as one integrated feature after this gate.
The user does not need to interact with partially activated variants.

## Comparator Consequences

The new stack changes the optimizer and therefore defines a native SNAKE stack,
not a selector-only ablation.

Future evidence should distinguish:

1. matched-backbone comparisons using one common optimizer;
2. native-stack comparisons:
   - SNAKE with integrated geometric recycling;
   - Geo-ADAPT with its native QNG selection and QNG refit;
   - append-only ADAPT with a strong conventional refit optimizer.

All metric, gradient, Hessian, refresh, and refit primitives must be counted.

## Explicit Exclusions

This integrated implementation does not include:

- candidate Jacobi-curvature lookahead;
- full Riemann-tensor reconstruction;
- Phase-II mixed candidate--ansatz Hessian expansion;
- manuscript edits;
- replacement or deletion of Powell evidence;
- default promotion of the new profile.

Those exclusions prevent the integrated optimizer project from becoming a
simultaneous rewrite of selection, scoring, and future-lookahead semantics.

## Implementation Verdict

An AI coding agent can implement this integrated stack, but it is not an easy
optimizer plug-in. The formal note now removes the central tensor ambiguity:
qBroyden predicts coordinate pullback stretch, while Riemannian BFGS recycles
inverse raised objective curvature in an orthonormal physical tangent frame.
The hard work is faithful quotient construction, endpoint transport, rank
transitions, and measurement accounting.

The contract is sufficiently concrete for implementation if the target is:

- exact-state/noiseless correctness first;
- measurement-compatible interfaces and accounting in the same integrated
  feature;
- qBroyden explicitly treated as an objective-dependent predictor between true
  geometric anchors;
- no claim that sparse hardware refresh is already statistically certified.

A fully calibrated hardware geometry filter remains research-grade because the
observation and uncertainty model has not yet been proved. That limitation does
not block implementing the complete exact-state geometric-recycling stack with
honest telemetry and interfaces.
