# Integrated Geometric Recycling Implementation Handoff

**Date:** 2026-07-11  
**Target:** a dedicated repo implementation agent  
**Action:** implement and verify one complete experimental optimizer feature  
**Do not:** edit the manuscript, launch scientific runs, alter canonical
defaults, or expose a partially implemented route

## Objective

Implement the complete exact-state/noiseless geometric-recycling optimizer
defined by the formal warm-start note and implementation contract. The feature
must combine, in one internally consistent state:

1. exact zero-coordinate Gram reuse at singleton or batch ansatz growth;
2. exact retained tangent-projector enlargement;
3. a labeled qBroyden coordinate-pullback predictor between true geometric
   anchors;
4. endpoint tangent-frame transport;
5. Riemannian inverse-BFGS objective-curvature recycling;
6. an explicitly typed direct-SR1 trust-region diagnostic branch;
7. singleton/batch insertion, pruning, rollback, branching, checkpoint, and
   resume remapping;
8. deterministic correction, refresh, and invalidation;
9. complete query/provenance accounting;
10. all mathematical, state, integration, and compatibility tests.

This is one atomic experimental feature. Internal modules and tests may be
implemented incrementally, but no partial optimizer key, CLI surface, or
Paper-I profile may become runnable until the complete acceptance gate passes.

## Governing Sources

Read these in order:

1. `AGENTS.md`
2. `MATH/AGENTS.md`
3. `MATH/paper_facing/README.md`
4. `MATH/paper_facing/paper_I_static_scaffold/paper_i_formal_manifold_warm_start_20260711.md`
5. `MATH/paper_facing/paper_I_static_scaffold/paper_i_integrated_geometric_recycling_implementation_contract_20260711.md`
6. this handoff
7. the current code and tests named below

The formal warm-start note is the mathematical authority. The shorter
implementation contract is the architecture and acceptance authority. If this
handoff conflicts with either, stop and follow the formal note first, then the
implementation contract.

The broader file
`paper_i_evolving_geometry_propagation_research_notes_20260711.md` is context,
not an implementation authority.

Do not edit `MATH/paper_details/Paper_I.tex` or regenerate `Paper_I.pdf` during
this task.

## Verified Live-Checkout State

The active checkout at handoff creation was:

```text
/Users/jakestrobel/local_repos/Holstein_test_fullclone_3
branch: codex/local-repo-health-20260706
HEAD: e1478cfbd6aa1792cd271c1c4f39b4d7f64ce8e8
```

Reverify this state before editing because it may have changed.

The following relevant paths were already modified or untracked:

```text
M  pipelines/scaffold/hh_continuation_scoring.py
M  pipelines/scaffold/hh_continuation_types.py
M  pipelines/static_adapt/adapt_pipeline.py
M  pipelines/static_adapt/cli_config.py
M  pipelines/static_adapt/engine_support.py
M  pipelines/static_adapt/optimizer_routes.py
M  test/test_adapt_vqe_integration.py
M  test/test_static_adapt_optimizer_routes.py
?? pipelines/static_adapt/joint_step_warm_start.py
?? pipelines/static_adapt/paper_i_runner.py
?? test/test_static_adapt_joint_step_warm_start.py
?? test/test_static_adapt_paper_i_runner.py
?? MATH/paper_facing/paper_I_static_scaffold/paper_i_evolving_geometry_propagation_research_notes_20260711.md
?? MATH/paper_facing/paper_I_static_scaffold/paper_i_formal_manifold_warm_start_20260711.md
?? MATH/paper_facing/paper_I_static_scaffold/paper_i_integrated_geometric_recycling_implementation_contract_20260711.md
?? MATH/paper_facing/paper_I_static_scaffold/paper_i_integrated_geometric_recycling_implementation_handoff_20260711.md
```

The eight tracked dirty paths represented approximately 9,240 insertions and
1,420 deletions when this handoff was created. Treat all of that work as
intentional user work. Do not reset, revert, overwrite, merge, switch branches,
stage, or commit it without current user approval.

Before editing:

1. run path-limited status and diffs for every file you will touch;
2. inspect the untracked files directly;
3. identify overlapping hunks;
4. preserve the current behavior and tests;
5. do not stop or alter unrelated processes or scientific jobs.

## Existing Warm Starts That Must Remain Distinct

Two existing modules use the phrase "warm start" but do not implement the new
persistent manifold state.

### Joint-selector seed

`pipelines/static_adapt/joint_step_warm_start.py` maps an already-computed joint
selector step into an optimizer `x0` and objective-guards it. It is an
initialization proposal only.

### Schur seed

`pipelines/static_adapt/schur_warm_start.py` builds append/prune seed proposals
from existing Schur response data. It is also an initialization proposal only.

Do not rename, absorb, or reinterpret either module as geometric recycling.
The first integrated geometric profile must reject simultaneous activation of
either seed mechanism because those seeds move the initial point before the
exact zero-coordinate geometric anchor. A future explicitly derived
composition may reanchor geometry after such a seed, but that is outside this
implementation.

The existing `paper_i_runner.py` may enable the joint-selector seed for another
funnel mode. Preserve that behavior. The new experimental geometric profile
must select its own mutually compatible settings without changing canonical
defaults.

## Binding Mathematical Contract

### Quotient frame

At every valid fixed-rank state, represent the retained tangent map as

\[
T_{\rho}=\mathcal E L,
\qquad
\mathcal E^{*_{\mathbb R}}\mathcal E=I_{\rho},
\qquad
G_{\rho}=L^{\mathsf T}L.
\]

The authoritative physical geometry is the orthonormal frame \(\mathcal E\)
and coordinate-to-frame map \(L\). Derive rather than independently persist:

\[
G=L^{\mathsf T}L,
\qquad
G^+=L^+L^{+\mathsf T},
\qquad
\Pi=\mathcal E\mathcal E^{*_{\mathbb R}},
\qquad
\bar r=L^{+\mathsf T}b.
\]

Use one explicit spectral-rank rule. A rank or retained-range change is a
structural transition, not an ordinary quasi-Newton update.

### Objective-curvature type

Keep these distinct:

\[
Q_{ij}=\partial_i\partial_jE,
\qquad
h_{ij}=\partial_i\partial_jE-\Gamma^a{}_{ij}\partial_aE,
\qquad
\mathcal A=\sharp_g\circ h^{\flat}.
\]

The positive-curvature branch stores

\[
B\approx\mathcal A^{-1}
\]

in the orthonormal physical frame. The step is exactly

\[
\boxed{
z=-B L^{+\mathsf T}b,
\qquad
p=L^+z.
}
\]

Do not apply a second inverse metric. Do not store an untyped object called
`hessian` whose variance is ambiguous.

The indefinite diagnostic branch stores a direct self-adjoint operator
\(A\approx\mathcal A\) and solves a trust-region SR1 model. Direct and inverse
branches are mutually exclusive authoritative modes.

### Exact zero-coordinate growth

For singleton or batch insertion at zero coordinates:

\[
G^+
=
\begin{bmatrix}
G_{AA}&G_{AB}\\
G_{BA}&G_{BB}
\end{bmatrix},
\qquad
G_{AA}=G^-.
\]

With new tangent frame \(D_B\), inherited projector \(\Pi_A\), and

\[
R_B=(I-\Pi_A)D_B,
\qquad
S_B=G_{BB}-G_{BA}G_{AA}^+G_{AB},
\]

the retained projector update is

\[
\Pi_{A\oplus B}
=
\Pi_A+R_BS_B^+R_B^{*_{\mathbb R}}.
\]

The old ordinary coordinate derivatives are exact under the neighborhood
restriction and explicit logical remapping. The old covariant-Hessian block is
not generally exact:

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

The enlarged inverse-curvature initialization

\[
B^+_0=\operatorname{diag}(B,\beta I)
\]

is a tagged regularized prior. Its mixed zero block is unknown, not measured or
exact.

### qBroyden predictor

Published qBroyden uses

\[
\mathsf F_{j+1}
=
(1-\epsilon_j)\mathsf F_j
+
\epsilon_jb_jb_j^{\mathsf T}.
\]

Because \(\mathsf F=4G\), the direct \(G\)-convention update is

\[
\widehat G_{j+1}
=
(1-\epsilon_j)\widehat G_j
+
\frac{\epsilon_j}{4}b_jb_j^{\mathsf T}.
\]

Implement the equivalent Sherman--Morrison inverse recurrence on one fixed
positive-definite quotient. It is not a Broyden secant and not intrinsic QGT
transport. It is objective and Hamiltonian-scale dependent. Every output must
label it `qbroyden_predicted`, never `measured_qgt` or `parallel_transport`.

qBroyden updates coordinate stretch only. It does not rotate the physical
tangent frame and does not change its identity metric. Do not include qBang
momentum.

### Endpoint transport

For equal-rank orthonormal endpoint frames, form

\[
C=\mathcal E_{j+1}^{*_{\mathbb R}}\mathcal E_j
=
U\Sigma V^{\mathsf T},
\qquad
Q=UV^{\mathsf T}.
\]

Use density tangents for gauge invariance, or prove and test an equivalent
phase-aligned horizontal-statevector representation. Transport physical-frame
vectors with \(Q\) and raised operators with \(QBQ^{\mathsf T}\).

Call this endpoint Procrustes transport. It is an isometric selected map and a
first-order approximation to Levi--Civita transport under the formal note's
conditions. Do not label it exact parallel transport. Same-state metric
matrices do not determine \(Q\).

### Riemannian secant update

Use the displacement actually accepted by globalization. In the endpoint
frame,

\[
s_j=Q_ja_j^{\rm acc},
\qquad
y_j=L_{j+1}^{+\mathsf T}b_{j+1}-Q_jL_j^{+\mathsf T}b_j.
\]

The inverse-BFGS branch must satisfy

\[
B_{j+1}y_j=s_j
\]

after Powell damping when required. The direct-SR1 branch must satisfy

\[
A_{j+1}s_j=y_j
\]

when its denominator guard passes. Never form a secant from a rejected trial
step or stale metric-raised gradient.

### Structural transitions

On rank change, spectral-gap closure, failed frame alignment, arbitrary remap
failure, prune that changes the physical state, or rollback:

- do not invert a rectangular transport;
- clear spanning secants;
- reset or conservatively rebuild qBroyden;
- restore the full checkpoint on rollback;
- treat compressed inverse curvature after prune only as a prior;
- reanchor geometry before another authoritative update.

## Required Architecture

Create an isolated package. Recommended ownership:

```text
pipelines/static_adapt/geometric_recycling/
  __init__.py
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

Keep tensor mathematics out of `adapt_pipeline.py`, `engine_support.py`, and
`paper_i_runner.py`.

### Configuration

Define one typed top-level configuration for the complete feature. It should
contain nested typed settings for:

- quotient rank resolution;
- exact anchor cadence;
- qBroyden predictor;
- inverse-BFGS versus direct-SR1 curvature mode;
- endpoint transport;
- correction/refresh thresholds;
- checkpoint serialization limits;
- query accounting.

Expose one nondefault experimental composition profile. Do not add every
numeric safeguard as a top-level CLI flag. Do not alter canonical or displayed
Paper-I settings.

Validate that geometric recycling is mutually exclusive with:

- `RouteAJointStepWarmStartConfig.enabled`;
- append/prune Schur seed warm-start modes;
- qBang/Adam momentum;
- any optimizer mode that cannot provide the required gradient and geometry
  interfaces.

### State

Implement a dedicated dense state. Do not put matrices into
`Phase2OptimizerMemoryAdapter`, whose remapping semantics are diagonal.

The state must include:

```text
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
metric_rank and retained spectrum
metric_anchor provenance
qBroyden reduced inverse and anchor age
authoritative curvature mode and matrix
trust radius
previous frame gradient
previous accepted step
refresh and query counters
invalidation reason
history tail
complete rollback checkpoint
```

Support:

- arbitrary singleton insertion;
- batch insertion;
- coordinate permutation;
- parameter tying where the existing layout explicitly supports it;
- prune/removal;
- branch-local clone isolation;
- rollback;
- bounded checkpoint serialization and resume validation.

### Providers

The optimizer must depend on explicit providers rather than closure access:

```text
energy(theta)
coordinate_differential(theta)
tangent_frame(theta) or tangent_action(theta, directions)
same_state_metric(theta, directions)
cross_tangent_gram(theta_new, theta_old) or transport_action(...)
optional_hessian_vector(theta, direction)
```

The exact-state backend may satisfy these from compiled state/tangent
propagation. Hardware-facing interfaces must not silently use statevectors.
Every provider call must expose query provenance.

### Optimizer dispatch

Build on the existing `AdaptInnerOptimizerDispatch` extraction in
`optimizer_routes.py`. Add one complete optimizer mode without changing the
behavior of `POWELL`, `BFGS`, `COBYLA`, `ROTOSOLVE`, `SPSA`, or `QNSPSA`.

Return the result shape expected by current callers, including at least:

```text
x
fun
nit
nfev
success
message
geometric_state or checkpoint reference
optimizer telemetry
query counters
```

Use the same integrated optimizer for ordinary inner refits, full refits,
boundary refits, and final refits when the profile is active. Do not quietly
fall back to Powell for one of those paths. A reset within the geometric
optimizer may use a conservative identity model, but must remain visible in
telemetry.

## Existing Files To Integrate Carefully

Inspect current live contents and dirty hunks before editing:

```text
pipelines/static_adapt/optimizer_routes.py
pipelines/static_adapt/paper_i_runner.py
pipelines/static_adapt/adapt_pipeline.py
pipelines/static_adapt/engine_support.py
pipelines/static_adapt/cli_config.py
pipelines/scaffold/hh_continuation_scoring.py
pipelines/scaffold/hh_continuation_types.py
pipelines/static_adapt/joint_step_warm_start.py
pipelines/static_adapt/schur_warm_start.py
```

Expected responsibilities:

- `optimizer_routes.py`: optimizer registration and typed dispatch only;
- `paper_i_runner.py`: one nondefault composition/config object and manifest
  translation;
- `adapt_pipeline.py`: construct providers, pass/remap state, preserve payloads;
- `engine_support.py`: optimizer key/result compatibility only;
- `cli_config.py`: compact experimental profile surface only if required;
- `hh_continuation_scoring.py`: expose existing tangent/Gram work through a
  narrow reusable workspace; do not duplicate score formulas;
- `hh_continuation_types.py`: shared protocol only if needed; do not overload
  diagonal optimizer memory;
- existing seed warm-start modules: remain unchanged except explicit
  compatibility validation if necessary.

Do not add candidate--active mixed objective-Hessian measurements to Phase II.
Do not change selection, batching, beam survival, prune authority, or Paper-I
score equations.

## Query And Provenance Accounting

Record separately:

\[
N_E,
\qquad
N_{\nabla E},
\qquad
N_G,
\qquad
N_{Hv},
\qquad
N_{\rm cross}.
\]

Requirements:

- classical qBroyden/BFGS/SR1/factorization operations cost no quantum query;
- exact reused old blocks are charged once;
- different physical states are different primitives;
- qBroyden gradient outer products reuse the already charged gradient;
- endpoint cross-frame observations are charged and labeled;
- optional HVPs are charged separately;
- all refits and final refits remain in total optimizer evaluation accounting;
- beam accounting follows the existing winning-branch Paper-I convention;
- statevector-only shortcuts are identified explicitly.

Do not use raw controller shot proxy as a substitute for enriched query work.

## Tests

Implement every one of the 26 falsification tests in
`paper_i_formal_manifold_warm_start_20260711.md`. Organize them into focused
files, for example:

```text
test/test_static_adapt_geometric_recycling_math.py
test/test_static_adapt_geometric_recycling_state.py
test/test_static_adapt_geometric_recycling_optimizer.py
```

Also extend, without replacing current coverage:

```text
test/test_static_adapt_optimizer_routes.py
test/test_static_adapt_paper_i_runner.py
test/test_adapt_vqe_integration.py
test/test_static_adapt_joint_step_warm_start.py
```

Minimum integration proofs:

1. profile-off behavior is unchanged;
2. the new profile rejects conflicting seed warm starts;
3. singleton and batch growth remap state correctly;
4. the optimizer applies the metric once;
5. qBroyden direct/inverse formulas agree numerically, including the factor of
   four;
6. zero-gradient and Hamiltonian-rescaling tests prove qBroyden is not labeled
   geometric truth;
7. endpoint Procrustes transport is gauge invariant and fails closed on rank or
   alignment loss;
8. inverse-BFGS and direct-SR1 satisfy their respective secants;
9. accepted rather than trial displacement is used;
10. prune compression is labeled a prior;
11. rollback restores every state field;
12. malformed checkpoint state fails closed;
13. query accounting changes exactly with anchor/transport/HVP calls;
14. Powell and every existing optimizer remain behaviorally unchanged;
15. one deterministic end-to-end exact-state fixture completes admission,
    full refit, optional prune/no-prune, checkpoint, resume, and final refit.

## Verification Commands

Use the repository's configured Python environment. At minimum run:

```bash
python -m py_compile \
  pipelines/static_adapt/geometric_recycling/*.py \
  pipelines/static_adapt/optimizer_routes.py \
  pipelines/static_adapt/paper_i_runner.py

pytest -q \
  test/test_static_adapt_geometric_recycling_math.py \
  test/test_static_adapt_geometric_recycling_state.py \
  test/test_static_adapt_geometric_recycling_optimizer.py \
  test/test_static_adapt_optimizer_routes.py \
  test/test_static_adapt_paper_i_runner.py \
  test/test_static_adapt_joint_step_warm_start.py

pytest -q \
  test/test_adapt_vqe_integration.py \
  test/test_static_adapt_route_a_funnel.py
```

If the local environment uses a different established invocation, use it and
record the exact command. Repair implementation failures and rerun the same
tests. Do not change scientific settings merely to make a test pass.

Do not launch Paper-I scientific runs in this implementation slice.

## Atomic Acceptance Gate

Do not expose the profile until all conditions pass:

1. all 26 mathematical falsification tests pass;
2. all remapping, integration, and compatibility tests pass;
3. existing optimizer behavior is unchanged when the profile is off;
4. the deterministic end-to-end fixture passes;
5. qBroyden is always labeled predicted and objective dependent;
6. the metric is applied exactly once;
7. no qBang momentum is active;
8. no Riemann/Jacobi lookahead is active;
9. no Phase-II mixed energy-Hessian expansion is active;
10. statevector-only and measurement-compatible paths are distinct;
11. query accounting reconciles anchors, reused blocks, cross-frame
    observations, gradients, HVPs, and optimizer evaluations;
12. canonical and displayed-results profiles are unchanged;
13. the new feature remains experimental and nondefault;
14. no runtime code was added to `Paper_I.tex` or manuscript support.

## Completion Report

Return:

1. exact files changed;
2. public types and optimizer key added;
3. how current dirty work was preserved;
4. mathematical invariants implemented;
5. tests and exact results;
6. deterministic fixture telemetry, including rank, anchors, resets, secants,
   and query categories;
7. remaining hardware-certification limitations;
8. any blocked acceptance item.

Do not claim paper promotion, manuscript readiness, or hardware certification.
Do not commit, stage, push, or create a branch unless the user explicitly asks.

## Files To Edit

Expected new files:

```text
pipelines/static_adapt/geometric_recycling/__init__.py
pipelines/static_adapt/geometric_recycling/config.py
pipelines/static_adapt/geometric_recycling/state.py
pipelines/static_adapt/geometric_recycling/quotient_frame.py
pipelines/static_adapt/geometric_recycling/pullback_predictor.py
pipelines/static_adapt/geometric_recycling/objective_curvature.py
pipelines/static_adapt/geometric_recycling/tangent_transport.py
pipelines/static_adapt/geometric_recycling/refresh_policy.py
pipelines/static_adapt/geometric_recycling/optimizer.py
pipelines/static_adapt/geometric_recycling/query_accounting.py
test/test_static_adapt_geometric_recycling_math.py
test/test_static_adapt_geometric_recycling_state.py
test/test_static_adapt_geometric_recycling_optimizer.py
```

Likely integration files, only after inspecting current dirty hunks:

```text
pipelines/static_adapt/optimizer_routes.py
pipelines/static_adapt/paper_i_runner.py
pipelines/static_adapt/adapt_pipeline.py
pipelines/static_adapt/engine_support.py
pipelines/static_adapt/cli_config.py
pipelines/scaffold/hh_continuation_scoring.py
pipelines/scaffold/hh_continuation_types.py
test/test_static_adapt_optimizer_routes.py
test/test_static_adapt_paper_i_runner.py
test/test_adapt_vqe_integration.py
test/test_static_adapt_joint_step_warm_start.py
```

Files explicitly out of scope:

```text
MATH/paper_details/Paper_I.tex
Paper_I.pdf
legacy route modules
scientific run artifacts
```
