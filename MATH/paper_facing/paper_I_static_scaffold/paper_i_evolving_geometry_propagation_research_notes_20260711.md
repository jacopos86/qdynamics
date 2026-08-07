# Evolving Geometry Propagation and Recycling Research Notes

**Date:** 2026-07-11  
**Status:** conceptual source-of-truth note; no code or manuscript changes authorized  
**Scope:** evolving tangent geometry, measurement reuse, parallel transport,
metric/Hessian recycling, uncertainty, and decision-preserving remeasurement

## Purpose

This note preserves the mathematical ideas developed in the side conversation
before that chat ends. It separates:

1. established Paper-I Phase-II mathematics;
2. corrections to discarded proposals;
3. general evolving-manifold ideas;
4. literature-backed conclusions from the first deep-research response;
5. open research questions about recycling the Gram matrix and Hessian;
6. the conditions required before any implementation is considered.

This is not a runtime-settings document, implementation handoff, manuscript
revision, or declaration of a new canonical route.

## Fast Reading Path

For a short pass, read these sections in order:

1. **Formal Warm-Start Resolution** for the completed quotient-frame result.
2. **Status Ledger** for what is established, rejected, or still speculative.
3. **Correction: Why the Proposed Joint Phase-II Replacement Was Wrong** for
   the query-cost correction.
4. **Actionable Exact Update at Ansatz Growth** for the exact Gram-block and
   tangent-projector recycling result.
5. **How Geometry Prediction Could Save Measurements** for the
   decision-certificate mechanism.
6. **Proposed Geometric Broyden Extension** for the predictive component after
   reoptimization.
7. **Open Questions** for the unresolved research program.

## Formal Warm-Start Resolution

The optimizer warm-start mathematics is now resolved in
`paper_i_formal_manifold_warm_start_20260711.md`, with the concise code-facing
contract in
`paper_i_integrated_geometric_recycling_implementation_contract_20260711.md`.
Those two files supersede tentative optimizer formulas in this research
notebook.

The resolved representation is

\[
T_{\rho}=\mathcal E L,
\qquad
G_{\rho}=L^{\mathsf T}L,
\]

with \(\mathcal E\) an orthonormal physical tangent frame. qBroyden predicts
the coordinate pullback between true geometric anchors, while Riemannian BFGS
recycles the inverse raised objective-Hessian operator \(B\) in the physical
frame. The type-correct step is

\[
z=-B L^{+\mathsf T}b,
\qquad
p=L^+z.
\]

Published qBroyden is the objective-gradient outer-product recurrence, not a
QGT secant update. It is therefore a labeled objective-dependent predictor,
not intrinsic metric transport. Riemann/Jacobi curvature is not required for
the optimizer warm start.

## Status Ledger

| Item | Status |
|---|---|
| Paper-I Phase-II novelty score | Established and should be preserved unless deliberately ablated |
| Moving mixed candidate--ansatz Hessian coupling into Phase II | Rejected; increases query burden |
| Parallel transport of tangent information | Mathematically established in fixed smooth regimes |
| Evolving-manifold predict--correct architecture | Research proposal assembled from established components |
| Decision-aware sparse remeasurement | Strong literature-supported direction; application-specific bounds still needed |
| Full Riemann tensor propagation | Rejected as a default; likely too expensive and unnecessary |
| Metric/Gram recycling | Exact zero-growth reuse resolved; qBroyden retained only as a labeled objective-dependent predictor between anchors |
| Objective-Hessian recycling | Formal Riemannian BFGS/SR1 translation resolved for the exact-state experimental stack |
| Joint Gram--Hessian recycling | Resolved through the quotient frame (T_{\rho}=\mathcal E L); the objects remain distinct |
| qBroyden plus objective-Hessian recycling | Type-compatible only through (z=-B L^{+\mathsf T}b), (p=L^+z); no second metric application |
| Candidate Jacobi-curvature lookahead | Speculative later selector route; not established Paper-I behavior |
| Code implementation | Not authorized in this note |
| `Paper_I.tex` modification | Not authorized in this note |

## Core Paper-I Phase-II Mathematics

At adaptive iteration (k), let

\[
|\psi_k\rangle
=
U_k(\boldsymbol\theta_k)|\phi_0\rangle.
\]

For active ansatz coordinates (i=1,\ldots,d_k), define the horizontal
Fubini--Study tangents

\[
|t_i\rangle
=
\Pi_k\partial_i|\psi_k\rangle,
\qquad
\Pi_k
=
I-|\psi_k\rangle\langle\psi_k|.
\]

For candidate record (r), define

\[
|t_r\rangle
=
\Pi_k\partial_{\alpha_r}|\psi_k(\alpha_r)\rangle
\big|_{\alpha_r=0}.
\]

The active Gram matrix, candidate--active overlap vector, and candidate norm are

\[
G_{ij}
=
\Re\langle t_i|t_j\rangle,
\qquad
q_i(r)
=
\Re\langle t_i|t_r\rangle,
\qquad
F(r)
=
\Re\langle t_r|t_r\rangle.
\]

The least-squares tangent novelty is

\[
\mathcal N_2(r)
=
1-
\frac{q(r)^{\top}G^+q(r)}{F(r)}.
\]

If

\[
a(r)=G^+q(r),
\]

then the residual tangent is

\[
|t_{\perp}(r)\rangle
=
|t_r\rangle
-
\sum_i a_i(r)|t_i\rangle,
\]

and

\[
\|t_{\perp}(r)\|^2
=
F(r)\mathcal N_2(r).
\]

The Phase-II directional energy model uses the candidate gradient (g(r)),
candidate directional curvature (h(r)), and Fubini--Study trust radius

\[
\Delta E_2(r)
=
\max_{\substack{\alpha\ge0\\
\alpha\sqrt{F(r)}\le\rho}}
\left[
g(r)\alpha
-
\frac12[h(r)]_+\alpha^2
\right].
\]

The Paper-I score family is

\[
S_2(r)
=
\frac{
\Delta E_2(r)\mathcal N_2(r)
}{
1+K(r)
}.
\]

The novelty multiplier is useful because it supplies an economical geometric
redundancy penalty without requiring mixed candidate--ansatz energy Hessian
measurements.

## Correction: Why the Proposed Joint Phase-II Replacement Was Wrong

An enlarged candidate--ansatz energy Hessian would be

\[
H_r
=
\begin{bmatrix}
H & c(r)\\
c(r)^{\top} & h(r)
\end{bmatrix},
\]

where

\[
c_i(r)
=
\left.
\frac{\partial^2E}
{\partial\theta_i\partial\alpha_r}
\right|_0.
\]

The vector (c(r)) is not generally recoverable as a classical dot product of
already measured (G,q,F,h), gradients, or the active Hessian. It contains new
mixed energy-derivative information.

For active width (w) and (L) candidates,

\[
\text{mixed-Hessian work}
=
O(wL).
\]

Candidate--candidate blocks used by a fully joint search can add

\[
O(L^2)
\]

work.

Therefore, moving (c(r)) into broad Phase-II scoring is a heavier route. The
query-efficient structure is:

1. preserve Paper-I Phase II using (g,h,F,q,G);
2. shortlist candidates;
3. measure (c(r)) only for a much smaller later joint singleton/batch search
   population when that route is explicitly used.

## Why the Gram Geometry Relates to Energy but Does Not Determine Curvature

For a tangent (|t\rangle), the first energy derivative is

\[
\frac{dE}{d\alpha}
=
2\Re
\langle t|(\widehat H-E)|\psi\rangle.
\]

Thus energy gradient is a linear functional on the state tangent space.

After metric projection,

\[
g_{\perp}(r)
=
g(r)-a(r)^{\top}g_A,
\]

where (g_A) is the active-coordinate gradient. Near an optimized ansatz,

\[
g_A\approx0,
\qquad
g_{\perp}(r)\approx g(r).
\]

The metric therefore supports an exact first-order state-space descent
quantity such as

\[
D_{\perp}(r)
=
\frac{g_{\perp}(r)^2}{F(r)\mathcal N_2(r)}
\]

when the residual metric is nonzero and sufficiently resolved.

However, the exact energy curvature along the projected path is

\[
h_{\perp}(r)
=
h(r)
-
2a(r)^{\top}c(r)
+
a(r)^{\top}Ha(r).
\]

The mixed vector (c(r)) is unavoidable in the exact projected second-order
energy model. Consequently:

- (G,q,F) determine projected state geometry;
- gradients determine first-order energy variation on that geometry;
- they do not determine the exact projected second-order energy curvature;
- the Phase-II novelty multiplier remains a useful low-query hybrid model.

## Metric, Objective Hessian, and Riemann Curvature

The three objects play different roles.

### Fubini--Study Gram matrix

\[
G_{ij}
=
\Re\langle t_i|t_j\rangle.
\]

It is positive semidefinite and defines local lengths and angles:

\[
ds^2
=
d\boldsymbol\theta^{\top}Gd\boldsymbol\theta.
\]

Although it is built from first state derivatives, it is the coefficient of the
second-order local expansion of state distance or infidelity.

### Objective Hessian

\[
H_{ij}
=
\frac{\partial^2E}
{\partial\theta_i\partial\theta_j}.
\]

It describes curvature of the scalar energy landscape over the manifold. It may
be indefinite and is not generally a metric.

### Riemann curvature

The Riemann tensor

\[
R_{ijkl}
\]

is derived from the metric field and its derivatives. It describes how tangent
geometry changes over the manifold and how parallel transport depends on path.
It is not a more detailed metric and is not the objective Hessian.

For two directions (u,v), sectional curvature is

\[
\mathcal K(u,v)
=
\frac{R(u,v,v,u)}
{\langle u,u\rangle_G\langle v,v\rangle_G
-
\langle u,v\rangle_G^2}.
\]

Intrinsic curvature becomes informative in at least two dimensions. It is more
relevant to finite-step candidate interactions, batching, and transport error
than to one isolated singleton direction.

The full tensor scales poorly and should not be a default persistent object.
Directional curvature bounds, second fundamental forms, or empirical transport
residuals are more plausible.

## Parallel Transport

At iteration (k), tangent vectors live in

\[
T_{\psi_k}\mathcal M_k.
\]

After the state moves, the old vectors do not automatically belong to

\[
T_{\psi_{k+1}}\mathcal M_{k+1}.
\]

Parallel transport supplies a map

\[
\mathcal P_{k\to k+1}:
T_{\psi_k}\mathcal M_k
\longrightarrow
T_{\psi_{k+1}}\mathcal M_{k+1}.
\]

Along a fixed smooth manifold, (v(t)) is parallel when

\[
\frac{Dv^i}{dt}
=
\frac{dv^i}{dt}
+
\Gamma^i_{ja}
\dot\theta^jv^a
=
0.
\]

Levi--Civita transport preserves inner products:

\[
\frac{d}{dt}
g(v,w)
=
0.
\]

This means the intrinsic metric is compatible with transport. Apparent changes
in coordinate Gram matrices can still arise from frame changes, tangent-subspace
changes, pullback-map changes, embeddings, and structural ansatz growth.

The deep-research result strongly recommends one transport rule per structural
regime. Projection transport, intrinsic transport, and curvature correction
should not be stacked as if they were independent physical effects.

## Evolving-Manifold Predict--Correct Architecture

The first deep-research response found no single mature framework covering the
entire problem. It recommended a synthesis of established components.

The smallest general state proposed there was schematically

\[
\Xi_k
=
(x_k,\Pi_k,\mathcal G_k,s_k,\mathcal U_k,\mathcal C_k),
\]

where:

- (x_k) is the current manifold point;
- (Pi_k) is an active tangent projector or subspace representation;
- (mathcal G_k) is the metric restricted to that active representation;
- (s_k) identifies the current fixed-dimension or fixed-rank regime;
- (mathcal U_k) represents uncertainty;
- (mathcal C_k) stores downstream decision certificates.

This notation deliberately avoids using (H_k) for the metric because (H)
already denotes the energy Hessian in the Paper-I context.

An important unresolved representation issue remains. If (Pi_k) is the full
Hilbert-space orthogonal projector and the ambient Fubini--Study inner product is
fixed, the induced metric on its image is already determined. Carrying both
(Pi_k) and an unrelated metric operator may duplicate information. A more
appropriate application representation may be either:

\[
\text{transported tangent frame }T_k,
\]

or

\[
\text{projector }\Pi_k
+
\text{parameter-to-tangent coordinate map}.
\]

This must be resolved before implementation.

## How Geometry Prediction Could Save Measurements

The central rule is

\[
\boxed{
\text{prediction schedules measurements; validated information controls decisions}
}.
\]

At iteration (k+1):

1. transport the previous tangent representation;
2. predict the new active Gram geometry;
3. measure selected anchor quantities;
4. compute innovations between prediction and observation;
5. build valid uncertainty intervals for decision-relevant scores;
6. measure additional quantities only while candidate ordering remains
   ambiguous;
7. either certify the decision or declare the geometric model obsolete.

For a general score (Phi(r)), maintain

\[
\Phi^-(r)
\le
\Phi(r)
\le
\Phi^+(r).
\]

A top-(m) shortlist (S) is certified when

\[
\min_{i\in S}\Phi^-(i)
>
\max_{j\notin S}\Phi^+(j).
\]

The observation loop is

\[
\text{observe until}
\quad
\left[
\text{decision certified}
\right]
\quad\lor\quad
\left[
\text{model obsolete}
\right].
\]

This is the only credible source of measurement savings. Transport and
curvature alone save nothing if every matrix element is still remeasured.

## Perturbation and Obsolescence Principles

If

\[
\widehat G
=
G+E,
\qquad
\|E\|_2\le\varepsilon,
\]

then quadratic forms satisfy

\[
\left|
r^{\top}\widehat Gr-r^{\top}Gr
\right|
\le
\varepsilon\|r\|_2^2.
\]

If (G\succeq\mu I) on the active subspace and (arepsilon<\mu), then

\[
\|\widehat G^{-1}-G^{-1}\|_2
\le
\frac{\varepsilon}{\mu(\mu-\varepsilon)}.
\]

For semidefinite metrics, useful pseudoinverse control also requires:

- a lower bound on the smallest retained nonzero eigenvalue;
- a nonclosing spectral gap;
- control of active-subspace rotation.

Principal-angle drift is naturally represented by projector difference:

\[
\|\widehat\Pi-\Pi\|_2
=
\|\sin\Theta(\widehat U,U)\|_2.
\]

The model should be retired or broadly refreshed after:

- spectral-gap closure;
- loss of the retained metric floor;
- rank transition;
- failed innovation or likelihood-ratio test;
- nonidentifiability under the selected observations;
- abrupt drift faster than the observation budget can resolve;
- structural ansatz change not represented by the current regime.

The major remaining gap is obtaining a valid bound

\[
\|\widehat G-G\|_2\le\varepsilon
\]

from sparse anchors. That requires assumptions such as bounded drift, low rank,
spectral separation, identifiable observations, or a calibrated stochastic
evolution model. Without such assumptions, uncertainty on unmeasured entries is
heuristic.

## Actionable Exact Update at Ansatz Growth

GPT-Pro identified one substantially more actionable result than the general
transport architecture: **the active Gram matrix has an exact block extension
at zero initialization of a newly admitted coordinate.**

Let the old ansatz be

\[
f_k:
\Theta_k
\longrightarrow
\mathbb P(\mathcal H),
\qquad
\boldsymbol\theta
\longmapsto
[\psi_k(\boldsymbol\theta)].
\]

Admit a new anti-Hermitian generator (A) with coordinate (alpha), and assume
the enlarged circuit satisfies

\[
|\psi_{k+1}(\boldsymbol\theta,0)\rangle
=
|\psi_k(\boldsymbol\theta)\rangle.
\]

This holds when the newly introduced unitary is the identity at
(alpha=0), and the inherited coordinates retain the same chart at that
insertion point. Then

\[
|t_i^{(k+1)}(\boldsymbol\theta,0)\rangle
=
|t_i^{(k)}(\boldsymbol\theta)\rangle.
\]

Define the new horizontal tangent

\[
|d_A\rangle
=
\Pi_kA|\psi_k\rangle.
\]

To avoid collision with the mixed **energy-Hessian** vector (c(r)), denote
the new metric-overlap vector by

\[
(q_A)_i
=
\Re\langle t_i^{(k)}|d_A\rangle,
\]

and its norm by

\[
F_A
=
\Re\langle d_A|d_A\rangle.
\]

The enlarged metric at the zero-initialized insertion point is exactly

\[
\boxed{
G_{k+1}(\boldsymbol\theta,0)
=
\begin{bmatrix}
G_k(\boldsymbol\theta) & q_A\\
q_A^{\top} & F_A
\end{bmatrix}
}.
\]

Therefore, if (G_k) is cached at the same point, the old--old block requires
no new measurements. The number of new independent scalar entries is

\[
d_k+1,
\]

rather than reconstructing all

\[
\frac{(d_k+1)(d_k+2)}{2}
\]

entries of the enlarged symmetric matrix.

The required new row and diagonal are precisely the same geometric objects
already represented by the candidate overlap (q_A) and norm (F_A). This is
an exact recycling statement, not a slowly-varying approximation.

### Exact tangent-projector extension

Let (Pi_k^{T}) denote the orthogonal projector onto the old active tangent
span. Decompose

\[
|d_A\rangle
=
\Pi_k^{T}|d_A\rangle
+
|r_A\rangle,
\qquad
|r_A\rangle
=
(I-\Pi_k^{T})|d_A\rangle.
\]

The residual norm is

\[
\sigma_A
=
\lVert r_A\rVert^2
=
F_A-q_A^{\top}G_k^+q_A.
\]

If (sigma_A>0), the active tangent projector has the exact rank-one update

\[
\boxed{
\Pi_{k+1}^{T}
=
\Pi_k^{T}
+
\frac{|r_A\rangle\langle r_A|_{\mathbb R}}
{\sigma_A}
}.
\]

If (sigma_A\approx0), the parameter count grows but the locally effective
tangent dimension does not. This is the same residual geometry measured by the
novelty construction.

### Exact versus predictive regimes

The distinction must remain explicit:

1. **At zero-initialized ansatz growth:** the block extension and projector
   update above are exact.
2. **After inner reoptimization moves the parameters:** the old Gram block is
   no longer exact and requires transport, prediction, and selective
   correction.
3. **At rank or structural transition:** the smooth fixed-regime model may fail
   and a broad refresh can be necessary.

This separates an immediately useful recycling rule from the more speculative
evolving-manifold filter.

## Proposed Geometric Broyden Extension

GPT-Pro proposed learning only the directional geometric evolution actually
encountered along optimizer steps, rather than measuring every Christoffel or
Riemann component.

Let

\[
\mathcal A_k[s]
\]

predict metric displacement for parameter step (s), and let

\[
\mathcal K_k[s]
\]

predict tangent-subspace displacement. If the fixed-rank metric and tangent
projector admit suitable manifold logarithms, define observed geometric secants

\[
Y_k
=
\operatorname{Log}_{G_k}(G_{k+1}),
\qquad
Z_k
=
\operatorname{Log}_{\Pi_k^{T}}(\Pi_{k+1}^{T}).
\]

A Broyden-style directional update is schematically

\[
\mathcal A_{k+1}[u]
=
\mathcal A_k[u]
+
\frac{s_k^{\top}u}{s_k^{\top}s_k}
\left(
Y_k-\mathcal A_k[s_k]
\right),
\]

with an analogous update for (mathcal K_k).

This is a **proposed synthesis**, not an established exact recycling identity.
It learns only contractions along observed optimizer directions. A single
trajectory does not identify the full connection, second fundamental form, or
curvature tensor.

The qBang/qBroyden literature supplies a partial precedent for Broyden updates
of a quantum Fisher information approximation, while Hessian-recycling work
supplies a precedent for growing and retaining inverse-Hessian information
across adaptive ansatz iterations. Neither source by itself establishes the
full tangent-projector predictor proposed here.

### Selective correction

The geometric Broyden prediction must remain a prior. Directional observations
can correct it through a constrained update such as

\[
G_{k+1}^{+}
=
\operatorname*{arg\,min}_{G\succeq0}
\left[
\lVert G-G_{k+1}^{-}\rVert_W^2
+
\sum_a
\frac{
\left(y_a-v_a^{\top}Gv_a\right)^2
}{
\sigma_a^2
}
\right].
\]

The observations $y_a$ may be selected directional quadratic forms. The
prediction is retired when innovation, spectral, rank, or decision-certificate
tests fail.

### Verified source anchors

- Stokes, Izaac, Killoran, and Carleo, *Quantum Natural Gradient*, establish
  that the real quantum geometric tensor is the pullback Fubini--Study metric:
  <https://arxiv.org/abs/1909.02108>.
- Ramoa, Santos, Mayhall, Barnes, and Economou, *Reducing measurement costs
  by recycling the Hessian in adaptive variational quantum algorithms*, recycle
  an evolving inverse-Hessian approximation across adaptive ansatz growth:
  <https://arxiv.org/abs/2401.05172> and
  <https://doi.org/10.1088/2058-9565/ad904e>.
- Fitzek, Jonsson, Dobrautz, and Schafer, *Optimizing Variational Quantum
  Algorithms with qBang*, use Broyden-style updates of a quantum Fisher
  information approximation:
  <https://arxiv.org/abs/2304.13882>.

## Compatibility of qBang, Hessian Recycling, and Manifold Warm Starts

These ideas are mathematically compatible only when they have different roles
inside one optimizer model.

### Metric role

qBang recycles an approximation to the inverse quantum Fisher or
Fubini--Study metric,

\[
\widehat G_k^{+},
\]

which supplies a natural-gradient direction

\[
\operatorname{grad}_{g}E
=
\widehat G_k^{+}\nabla E.
\]

It controls how a coordinate gradient is interpreted as a state-space
direction.

### Energy-curvature role

Objective-Hessian recycling carries an approximation to the covariant energy
Hessian or its inverse:

\[
\widehat{\mathcal H}_k
\approx
\operatorname{Hess}_{g}E.
\]

It controls the finite optimizer response to the natural gradient.

### Unified step

The coherent Riemannian Newton or quasi-Newton equation is

\[
\widehat{\mathcal H}_k[p_k]
=
-\operatorname{grad}_{g}E.
\]

The metric raises the gradient index and defines tangent geometry, while the
recycled energy Hessian approximates objective curvature as an operator on that
tangent space.

The two recycled objects must not independently generate competing update
directions. One should not naively multiply unrelated inverse preconditioners
such as

\[
\widehat G_k^{+}\widehat H_k^{-1}
\]

without deriving their tensor types and coordinate transformations. Such a
product can double-precondition a direction or violate metric
self-adjointness.

### Manifold warm-start role

The evolving-manifold model supplies the prior and transport:

\[
(\Pi_k^{T},G_k,\widehat{\mathcal H}_k)
\xrightarrow{\mathcal T_k}
(\Pi_{k+1}^{T,-},G_{k+1}^{-},
\widehat{\mathcal H}_{k+1}^{-}).
\]

Then:

- exact block extension initializes (G_{k+1}) at zero-amplitude ansatz
  growth;
- qBang-style updates approximate metric evolution during reoptimization;
- Hessian recycling approximates energy-curvature evolution;
- sparse metric and gradient observations correct both models;
- rank, innovation, and decision tests determine refresh.

The hierarchy is

\[
\boxed{
\text{transported manifold prior}
\longrightarrow
\begin{cases}
\text{metric correction},\\
\text{energy-curvature correction}
\end{cases}
\longrightarrow
\text{one optimizer step}
}.
\]

qBang should be the metric-update mechanism inside this stack, not an
additional metric predictor layered on top of a second Broyden metric model.

### Relationship to Powell

Powell is derivative-free, whereas qBang and Hessian recycling require gradient
or secant information. Therefore:

- replacing Powell by the unified geometric optimizer defines a new optimizer
  stack;
- retaining Powell while measuring gradients only to maintain geometric
  predictors can increase query cost;
- current Powell evidence must remain separate from any new
  geometric-optimizer evidence.

## Candidate Jacobi-Curvature Lookahead

GPT-Pro proposed a more ambitious use of curvature: forecast how admitting one
candidate changes the future value of the remaining candidates.

For a normalized residual candidate direction (u_\mu), define

\[
\mathcal J_\mu(w)
=
R(w,u_\mu)u_\mu,
\]

or its active-space matrix

\[
(\mathcal J_\mu)_{ij}
=
R(e_i,u_\mu,e_j,u_\mu).
\]

This Jacobi operator describes geodesic focusing, divergence, and rotation in
planes containing the hypothetical candidate direction. It is smaller than a
full Riemann tensor, but it is not query-neutral.

### Candidate-augmented geometry

The residual candidate direction generally lies outside the current active
tangent manifold. Its curvature must therefore be defined on a
candidate-augmented manifold such as

\[
\mathcal M_{k,\mu}
=
\left\{
e^{\alpha_\mu A_\mu}
U_k(\boldsymbol\theta)|\phi_0\rangle
\right\},
\]

or on a temporary pool-augmented manifold. Curvature intrinsic to
(mathcal M_k) alone cannot evaluate a direction outside its tangent space.

### Future-score structure

Along a hypothetical geodesic generated by (u_\mu), future candidate value
depends on both energy-field evolution and geometric deflection. Schematically,

\[
\frac{d^2}{dt^2}
\left[dE(V_\nu(t))\right]_{t=0}
\sim
(\operatorname{D}_{u_\mu}\operatorname{Hess}_{g}E)
(u_\mu,v_\nu)
-
dE\!\left(R(v_\nu,u_\mu)u_\mu\right),
\]

plus generator-field terms when

\[
\frac{DV_\nu}{dt}\ne0.
\]

The roles are distinct:

- (R) predicts geometric deflection of other candidate directions;
- (operatorname{Hess}_{g}E) predicts local energy curvature;
- its covariant derivative predicts change of the energy curvature;
- generator-field derivatives describe evolution not determined by intrinsic
  geometry.

The Riemann tensor alone cannot forecast future ADAPT scores.

### Measurement-conscious form

A possible later architecture is

\[
\text{broad gradient scan}
\longrightarrow
\text{metric/novelty shortlist}
\longrightarrow
\text{Jacobi lookahead on finalists}
\longrightarrow
\text{exact finalist evaluation}.
\]

For shortlist width (L), selected candidate-pair contractions form roughly an
(L\times L) interaction model rather than a full rank-four tensor. The route
may nevertheless require candidate-conditioned curvature, covariant
energy-Hessian derivatives, generator-vector-field derivatives, and additional
uncertainty measurements.

It must therefore remain a distinct later experimental route. It is not part of
the immediate qBang/Hessian-recycling optimizer stack and should not be mixed
into the current Phase-II score without a separate query analysis.

### Curvature recycling boundary

Within one fixed manifold, selected curvature contractions may be transported
and corrected. Across ansatz growth, old curvature is not automatically the
old--old block of the enlarged-manifold curvature. The Gauss equation introduces
the second fundamental form of the inclusion. Old curvature is a close prior
only when the enlarged direction makes the old manifold approximately totally
geodesic.

Curvature propagation across ansatz growth therefore also requires information
about extrinsic bending, tangent-projector derivatives, or direct
generator-field evolution.

## Isolation Architecture To Avoid Route Clutter

If implemented later, this work should not become additional closures or
conditionals inside the current mega-pipeline.

The recommended separation is:

1. The existing Route-A core remains unchanged.
2. A new experimental optimizer lane owns qBang, qBroyden, and recycled
   objective-Hessian state.
3. A geometry-recycling module owns tangent frames/projectors, exact Gram block
   extension, transport, sparse correction, and refresh policy.
4. A separate curvature-lookahead module remains absent or default-off until
   its measurement contract is justified.
5. One typed composition profile activates the experimental stack; avoid many
   independent closure variables.
6. Matched-backbone and native-stack comparators remain separate benchmark
   modes.

A possible conceptual boundary is

```text
geometry_recycling/
  metric_state.py
  tangent_transport.py
  refresh_policy.py
  energy_curvature_state.py
  qbang_optimizer.py
  jacobi_lookahead.py
```

These are conceptual names, not approved implementation paths.

The staged research order is:

1. exact Gram block extension;
2. qBang/qBroyden as an isolated optimizer lane;
3. objective-Hessian recycling in the same tangent coordinates;
4. transported manifold warm start and sparse refresh;
5. only then candidate Jacobi-curvature lookahead.

This order limits brittleness because every stage can be compared with the
unchanged current route before another mechanism is introduced.

## Hessian and Gram Recycling

### Objective-Hessian recycling

Established quasi-Newton and Hessian-recycling methods use transported secant
information. For step (s_k) and transported gradient difference (y_k), a
geometric secant condition has the form

\[
\widehat H_{k+1}s_k
\approx
y_k.
\]

One secant pair constrains only the Hessian action along observed directions. It
does not identify the full Hessian. Riemannian BFGS, SR1, multisecant,
limited-memory, Krylov, and preconditioner recycling are relevant literatures.

### Gram recycling

A possible metric prediction is

\[
\widehat G_{k+1}^{-}
=
A_k^{\top}G_kA_k,
\]

where (A_k) maps transported coordinates. However, exact metric-compatible
parallel transport preserves intrinsic inner products. Therefore, a Gram update
must distinguish:

- coordinate change;
- tangent-subspace change;
- change in the pullback map;
- structural manifold enlargement;
- numerical transport error.

It is not yet established whether the most useful application object is the
Gram matrix, tangent frame, tangent projector, low-rank factor, or a
decision-relevant generalized operator.

### Joint recycling

Potential decision-relevant joint objects include

\[
G^+H,
\qquad
Hv=\lambda Gv,
\qquad
G^+g,
\]

and metric-constrained trust-region models. Recycling their invariant subspaces
or matrix-vector actions may be more useful than reconstructing full (G) and
(H) independently.

## Recovering Higher Geometry

The information hierarchy is

\[
\text{state}
\rightarrow
\text{tangents}
\rightarrow
G
\rightarrow
\nabla G
\rightarrow
\Gamma
\rightarrow
R,
\]

while objective information follows

\[
E
\rightarrow
dE
\rightarrow
\operatorname{Hess}E
\rightarrow
\nabla\operatorname{Hess}E.
\]

These are different hierarchies. The Hessian of one scalar objective does not
determine the metric connection or Riemann curvature without additional
structure.

A single trajectory of recycled (G_k) values can generally estimate only
directional contractions such as

\[
\dot\theta^a\partial_aG.
\]

Recovering the full local derivative tensor requires independent probing
directions. Recovering curvature requires still richer information, such as
small loops, holonomy, geodesic deviation, multiple trajectory directions, or
second fundamental forms.

Changes in tangent projectors may reveal extrinsic bending more economically
than reconstructing the full intrinsic Riemann tensor. This should be a major
focus of the recycling research.

## Deep-Research Artifacts

The following standalone prompts preserve the two open research programs:

1. `prompt-exports/2026-07-11-evolving-variational-manifold-deep-research-gpt-pro-handoff.md`
   - general evolving, partially observed manifolds;
   - transport, sparse correction, decision certification, and obsolescence.

2. `prompt-exports/2026-07-11-hessian-gram-recycling-evolving-manifold-gpt-pro-handoff.md`
   - objective-Hessian recycling;
   - Gram/metric recycling;
   - joint recycling;
   - recoverability of higher geometric tensors;
   - observation-saving guarantees.

The first returned deep-research response identified the following strongest
foundations:

- geometric filtering in fixed smooth regimes;
- Grassmann/Stiefel subspace tracking;
- fixed-rank PSD geometry;
- dynamical low-rank approximation;
- perturbation theory for eigenspaces and pseudoinverses;
- fixed-confidence sequential decision methods;
- innovation and change-point tests.

Its principal conclusion was that no single established theory combines all of
these with changing dimension and decision-preserving sparse observation.

## Open Questions

1. What is the minimal nonredundant persistent state for a pullback
   Fubini--Study manifold?
2. Should the application track a tangent frame, tangent projector, coordinate
   map, low-rank metric factor, or some combination?
3. Which single transport rule is most appropriate within a fixed ansatz
   regime?
4. How should operator admission and dimension growth be represented: nested
   manifold, flag manifold, stratified transition, or explicit structural mode?
5. Which sparse anchor observations identify a valid bound on unmeasured Gram
   entries?
6. How should uncertainty in (G,q,F) be propagated into a rigorous interval
   for \(\mathcal N_2(r)\) and (S_2(r))?
7. Can a decision certificate save measurements without changing Paper-I score
   semantics?
8. Which objective-Hessian recycling method remains stable under the relevant
   metric and rank deficiencies?
9. Is it better to recycle (G) and (H), or generalized eigenspaces and
   matrix-vector actions derived from them?
10. Which higher geometric contractions can be estimated from accepted-step
    trajectories without prohibitive new measurements?
11. When is full refresh unavoidable?
12. What early-iteration versus late-iteration assumptions can be validated
    rather than imposed heuristically?

## Recommended Next Research Sequence

1. Complete the Hessian/Gram recycling deep research using the second export.
2. Resolve the persistent-state representation and transport rule.
3. Derive application-specific perturbation bounds for
   \(\mathcal N_2(r)\) and (S_2(r)).
4. Define sparse-anchor identifiability assumptions and full-refresh triggers.
5. Design deterministic numerical experiments comparing full refresh with
   transported partial refresh.
6. Only after those steps, consider a typed experimental implementation.

No implementation or manuscript promotion should follow directly from this
note without a separate behavioral decision.
