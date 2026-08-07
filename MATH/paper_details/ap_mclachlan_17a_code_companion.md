---
title: "17A AP-McLachlan Code Companion"
subtitle: "Execution-order guide from repo code to Chapter 17A symbols"
date: "2026-06-29"
geometry: margin=0.6in
documentclass: extarticle
fontsize: 9pt
classoption:
  - twocolumn
header-includes:
  - \setlength{\columnsep}{0.24in}
  - \setlength{\parskip}{1.5pt}
  - \setlength{\parindent}{0pt}
  - \AtBeginDocument{\fvset{fontsize=\scriptsize}}
  - \AtBeginDocument{\raggedright}
  - \AtBeginDocument{\setlength{\abovedisplayskip}{3pt plus 1pt minus 1pt}\setlength{\belowdisplayskip}{3pt plus 1pt minus 1pt}\setlength{\abovedisplayshortskip}{2pt plus 1pt minus 1pt}\setlength{\belowdisplayshortskip}{2pt plus 1pt minus 1pt}}
  - \sloppy
---

# High-Level Flow

```mermaid
flowchart TD
  A[Static ansatz artifact]
  B[Load ansatz input]
  C[AP state U_k(theta)]
  D[H(t_k)]
  E[Geometry K,f,V]
  F[theta_dot solve]
  G[Integrate theta]
  H[Trajectory point]
  I{Append?}
  J[Patch score]
  K[Commit append]
  L[Next time point]
  A --> B --> C
  C --> D --> E --> F --> G --> H --> I
  I -- no --> L
  I -- yes --> J --> K --> G
```

The same flow as repo paths:

```text
artifact JSON
  -> pipelines/scaffold/runtime_loader.py
       load_scaffold_runtime_input(...)
  -> pipelines/time_dynamics/ap_mclachlan/state.py
       state_from_scaffold_runtime_input(...)
       APMcLachlanState
  -> pipelines/time_dynamics/ap_mclachlan/hamiltonian.py
       time_dependent_hamiltonian_from_runtime_input(...)
       TimeDependentHamiltonian.matrix_at(...)
  -> pipelines/time_dynamics/ap_mclachlan/geometry_eval.py
       evaluate_mclachlan_geometry(...)
  -> pipelines/time_dynamics/ap_mclachlan/geometry.py
       McLachlanGeometry(K, f, norm_b_sq)
  -> pipelines/time_dynamics/ap_mclachlan/inverse.py
       supported_inverse(...), solve_theta_dot(...)
  -> pipelines/time_dynamics/ap_mclachlan/fixed_step.py
       solve_fixed_mclachlan_step(...)
  -> pipelines/time_dynamics/ap_mclachlan/integrators.py
       integrate_theta_step(...)
  -> pipelines/time_dynamics/ap_mclachlan/trajectory.py
       run_fixed_mclachlan_trajectory(...)
  -> pipelines/time_dynamics/ap_mclachlan/support_patch.py
       score_support_patch(...)
  -> pipelines/time_dynamics/ap_mclachlan/adaptive_trajectory.py
       _select_append_patch(...), run_append_mclachlan_trajectory(...)
```

External entry points:

```text
pipelines/time_dynamics/runners/ap_fixed_from_adapt_artifact.py
  run_fixed_ap_mclachlan_from_artifact(...)

pipelines/time_dynamics/runners/ap_append_from_adapt_artifact.py
  run_append_ap_mclachlan_from_runtime_input(...)
```

# Execution Overview

The route uses $\nu_k=(B_k^-,R_k^+)$. The repo does not begin by choosing
$\nu_k$. It first materializes a fixed-support McLachlan step, then later asks
whether the support should be edited.

The package prefix is `pipelines/time_dynamics/ap_mclachlan/`. The fresh package
currently implements the fixed-support primitives, Hamiltonian provider, state
adapter, geometry evaluator, integrators, fixed trajectory runner,
support-patch scorer, and an append-first adaptive trajectory path. The still
incomplete part is the full prune/exchange controller with nonlinear projection,
shadow checks, rollback, and cooldown.

The code-facing route is:

```text
static ansatz artifact
  -> ansatz runtime input
  -> APMcLachlanState
  -> TimeDependentHamiltonian
  -> evaluate_mclachlan_geometry
  -> solve_fixed_mclachlan_step
  -> integrate_theta_step
  -> run_fixed_mclachlan_trajectory
  -> score_support_patch
  -> run_append_mclachlan_trajectory
  -> later prune/exchange controller layer
```

The same route in Chapter 17A symbols is:

$$
\begin{aligned}
(U_k(\theta_k)|\psi_{\mathrm{ref}}\rangle,H(t_k))
&\to(\bar T_k,\bar b_k,K_k,f_k,V_k)\\
&\to \dot\theta_k=K_{k,\tau,\varepsilon}^{\oplus}f_k\\
&\to U_k(\theta_{k+1})|\psi_{\mathrm{ref}}\rangle
\to \nu_k=(B_k^-,R_k^+).
\end{aligned}
$$

The mental model is simple: the code prepares the current variational state,
builds a tangent least-squares problem, solves for the best tangent velocity,
integrates the parameters, and only then has enough local information to score
append/prune/exchange edits.

> **Implemented files.** `state.py`, `hamiltonian.py`, `inverse.py`,
> `geometry.py`, `geometry_eval.py`, `fixed_step.py`, `integrators.py`,
> `trajectory.py`, `support_patch.py`, and `adaptive_trajectory.py`.
>
> **Runner.**
>
> ```text
> pipelines/time_dynamics/runners/
>   ap_fixed_from_adapt_artifact.py
> ```
>
> **Controller boundary.** Append-first support editing exists. Full prune and
> exchange acceptance still require the nonlinear projection and safety protocol.

# Step 1. Load the Static Ansatz

The runner starts from a static ansatz artifact produced by ADAPT,
SNAKE, Geo-ADAPT, or another static route. The AP core does not directly reason
about the manuscript table or static route provenance; it consumes the shared
runtime contract:

$$
|\psi_k\rangle=U_k(\theta_k)|\psi_{\mathrm{ref}}\rangle .
$$

```text
U_k                 -> terms + layout
theta_k             -> theta_runtime
|psi_ref>           -> psi_ref
```

Code anchor:

```python
runtime_input = load_scaffold_runtime_input(...)
state = state_from_scaffold_runtime_input(runtime_input)
```

The code anchor above uses legacy loader names; read those names as
ansatz-runtime loader names.

> **File.** `state.py`
>
> **Objects.** `APMcLachlanState`, `state_from_scaffold_runtime_input(...)`,
> `load_ap_mclachlan_state(...)`.
>
> **Field map.**
>
> ```text
> terms, layout       -> U_k
> theta_runtime       -> theta_k
> psi_ref             -> |psi_ref>
> executor            -> U_k(theta), d_theta U_k(theta)
> ```

Transformation pseudo-code:

```python
def state_from_runtime_input(runtime_input):
    terms = runtime_input.selected_terms
    layout = runtime_input.base_layout

    theta = as_float_vector(runtime_input.theta_runtime)
    psi_ref = normalize(runtime_input.psi_ref)
    psi_initial = normalize(runtime_input.psi_initial)

    assert len(theta) == layout.runtime_parameter_count
    assert len(terms) == layout.logical_parameter_count

    executor = CompiledAnsatzExecutor(terms, layout)
    return APMcLachlanState(
        terms=terms,
        layout=layout,
        theta_runtime=theta,
        psi_ref=psi_ref,
        psi_initial=psi_initial,
        executor=executor,
        static_hamiltonian=runtime_input.h_poly,
        candidate_pool_terms=runtime_input.candidate_pool_terms,
    )
```

Pedagogical check:

$$
(U_k,\theta_k,|\psi_{\mathrm{ref}}\rangle)
\longrightarrow
(\bar T_k,K_k,f_k).
$$

# Step 2. Build the Time-Dependent Hamiltonian Provider

The Hamiltonian provider is family-neutral. Hubbard-Holstein details enter
before this layer through the resolved problem and drive adapter. Chapter 17A
may write $H(t)=H_{\mathrm{static}}+c(t)D$, but the implementation exposes the
more general operation: give me the Pauli polynomial or matrix for the requested
time point.

Code anchor:

```python
hamiltonian = time_dependent_hamiltonian_from_runtime_input(
    runtime_input,
    drive_config=drive_config,
)
H_t = hamiltonian.matrix_at(time)
```

> **File.** `hamiltonian.py`
>
> **Object.** `TimeDependentHamiltonian`.
>
> **Builder.**
>
> ```text
> time_dependent_hamiltonian_from_runtime_input(...)
> ```
>
> **Methods.** `polynomial_at(time)` forms $H(t)$ as a Pauli polynomial, and
> `matrix_at(time)` forms the dense matrix used by the current statevector
> evaluator.

Transformation pseudo-code:

```python
def matrix_at(t):
    H = clone(static_poly)

    if drive_model is not None:
        coeff = drive_model.coefficient_at(t)
        H = H + coeff * drive_model.drive_poly

    return hamiltonian_matrix(H)
```

Reader cue: the McLachlan layer does not need to know whether the time dependence
came from a Gaussian drive, a pulse table, or no drive. It only needs the current
$H(t_k)$ when building the drift direction.

# Step 3. Prepare the State and Tangent Columns

At time point $k$:

$$
|\psi_k\rangle=U_k(\theta_k)|\psi_{\mathrm{ref}}\rangle,\qquad
Q_{\psi_k}=I-|\psi_k\rangle\langle\psi_k|,
$$

$$
\bar T_k
=Q_{\psi_k}
\left[
\partial_{\theta_{k,1}}|\psi_k\rangle,\ldots,
\partial_{\theta_{k,N_k}}|\psi_k\rangle
\right].
$$

Code operation:

```python
tangent_horizontal = (
    tangent - psi * np.vdot(psi, tangent)
)
```

> **File.** `geometry_eval.py`
>
> **Function.** `evaluate_mclachlan_geometry(...)`.
>
> **Helper.** `_horizontalize(vector, psi=psi)`.

Geometry-preparation pseudo-code:

```python
theta = theta_runtime or state.theta_runtime
indices = runtime_indices or range(len(theta))

psi, tangents_by_index = (
    state.executor.prepare_state_with_runtime_tangents(
        theta,
        state.psi_ref,
        runtime_indices=indices,
    )
)
psi = normalize(psi)

Tbar_columns = []
for idx in indices:
    raw = tangents_by_index[idx]
    horizontal = raw - psi * inner(psi, raw)
    Tbar_columns.append(horizontal)
```

After this step:

```text
Tbar_columns -> bar T_k
```

# Step 4. Build the Horizontal Schrodinger Drift

The evaluator computes:

$$
H(t_k)|\psi_k\rangle,\qquad
E_k=\langle\psi_k|H(t_k)|\psi_k\rangle,\qquad
\bar b_k=-i(H(t_k)-E_kI)|\psi_k\rangle.
$$

Code anchor:

```python
h_psi = hmat @ psi
energy = real(vdot(psi, h_psi))
b_bar = -1j * (h_psi - energy * psi)
```

> **File.** `geometry_eval.py`
>
> **Local variable.** `b_bar`.

Code map:

```text
Tbar   -> bar T_k
b_bar  -> bar b_k
```

# Step 5. Assemble `K`, `f`, and `norm_b_sq`

Once $\bar T_k$ and $\bar b_k$ exist, the implementation forms
$K_k=\Re(\bar T_k^\dagger\bar T_k)$, $f_k=\Re(\bar T_k^\dagger\bar b_k)$, and
$V_k=\|\bar b_k\|_2^2$. The least-squares objective is
$\mathcal R(u)=\|\bar T_ku-\bar b_k\|_2^2=u^\top K_ku-2u^\top f_k+V_k$.

The code stores these in `McLachlanGeometry`:

```python
Tbar = column_stack(Tbar_columns)
K = real(Tbar.conj().T @ Tbar)
f = real(Tbar.conj().T @ b_bar)
V = real(inner(b_bar, b_bar))

geometry = McLachlanGeometry(
    K=0.5 * (K + K.T),
    f=f,
    norm_b_sq=V,
    support_indices=indices,
    support_labels=labels,
    time=time,
)
```

> **Files.** `geometry.py`, `geometry_eval.py`
>
> **Objects.** `McLachlanGeometry`, `GeometryEvaluation`.
>
> **Field map.**
>
> ```text
> K          -> K_k, G_k
> f          -> f_k
> norm_b_sq  -> V_k
> ```

# Step 6. Solve the Supported Inverse Problem

The formal normal equation is $K_k\dot\theta_k=f_k$. The implemented solve uses
the supported inverse $\dot\theta_k=K_{k,\tau,\varepsilon}^{\oplus}f_k$:
symmetrize the matrix, optionally apply ridge, diagonalize, keep supported
eigenmodes, and invert only those modes.

Code anchor:

```python
inverse = supported_inverse(K, policy=policy)
theta_dot = inverse.inverse @ f
gamma = f @ theta_dot
```

The explained drift is
$\Gamma_k^K(J_k)=f_k^\top K_{k,\tau,\varepsilon}^{\oplus}f_k$, stored through
the solve result as `gamma`.

> **File.** `inverse.py`
>
> **Objects.** `McLachlanInversePolicy`, `SupportedInverse`,
> `McLachlanSolve`.
>
> **Functions.** `supported_inverse(...)`, `solve_theta_dot(...)`,
> `gamma_for_support(...)`.

Transformation pseudo-code:

```python
def solve_theta_dot(K, f, policy):
    K_sym = 0.5 * (K + K.T)
    K_ridge = K_sym + policy.ridge_lambda * I

    eigvals, eigvecs = eigh(K_ridge)
    threshold = policy.pinv_rcond * max(abs(eigvals))
    keep = abs(eigvals) > threshold

    inv_eigs = zeros_like(eigvals)
    inv_eigs[keep] = 1.0 / eigvals[keep]
    K_plus = (eigvecs * inv_eigs) @ eigvecs.T

    theta_dot = K_plus @ f
    gamma = f @ theta_dot
    return theta_dot, gamma
```

The geometric intuition is: do not invert raw coordinates; invert only
physically resolved tangent modes. If two parameters move the state in nearly
the same direction, the supported inverse avoids pretending that the duplicate
coordinate is independent.

# Step 7. Report the Fixed McLachlan Step

`fixed_step.py` packages the result of the solve. The two values to track are
$\dot\theta_k=K_{k,\tau,\varepsilon}^{\oplus}f_k$ and
$\Gamma_k^K(J_k)=f_k^\top\dot\theta_k$. The residual is
$\epsilon_k^2=V_k-\Gamma_k^K(J_k)$, and the normalized residual ratio is
$\rho_k=\epsilon_k^2/(V_k+\varepsilon)$.

Code anchor:

```python
solve = solve_theta_dot(geometry.K, geometry.f, policy=inverse_policy)
residual_sq = max(0.0, geometry.norm_b_sq - solve.gamma)
residual_ratio = residual_sq / residual_denominator(...)
```

> **File.** `fixed_step.py`
>
> **Object.** `FixedMcLachlanStep`.
>
> **Function.** `solve_fixed_mclachlan_step(...)`.
>
> **Schema marker.** `equation_id = "eq8_fixed_support_mclachlan"`.

This is the smallest code unit corresponding to the Chapter 17A fixed-support
McLachlan equation. If you want to understand whether the current support is
good enough at one time point, this is the object to inspect.

# Step 8. Integrate Parameters to the Next Time Point

The solve gives a vector field value
$d\theta/dt=F_{U_k}(\theta,t)=K_{k,\tau,\varepsilon}^{\oplus}f_k$. An
integrator converts that velocity into a new parameter vector. Euler uses
$\theta_{n+1}=\theta_n+\Delta tF(\theta_n,t_n)$; RK4 uses
$\theta_{n+1}=\theta_n+\frac{\Delta t}{6}(k_1+2k_2+2k_3+k_4)$.

Code anchor:

```python
integration = integrate_theta_step(
    theta=theta_current,
    t=time_value,
    dt=dt,
    rhs=theta_dot_rhs,
    method=integrator_method,
)
theta_next = integration.theta_next
```

Actual branch structure:

```python
if method == "euler":
    k1 = rhs(theta, t)
    theta_next = theta + dt * k1

if method == "rk4":
    k1 = rhs(theta, t)
    k2 = rhs(theta + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = rhs(theta + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = rhs(theta + dt * k3, t + dt)
    theta_dot = (k1 + 2*k2 + 2*k3 + k4) / 6
    theta_next = theta + dt * theta_dot
```

> **File.** `integrators.py`
>
> **Function.** `integrate_theta_step(...)`.
>
> **Object.** `IntegrationStep`.

Important separation: the McLachlan solve chooses $\dot\theta$, while the
integrator chooses how $\theta$ advances. This distinction matters when reading
AP-McLachlan results: a bad step can come from the support geometry, the inverse
policy, the time step, or the integrator.

# Step 9. Run a Fixed-Support Trajectory

The fixed-support trajectory runner loops over the time grid. For each time
point it evaluates geometry, solves the McLachlan step, records diagnostics, and
integrates to the next point. Symbolically, this runner maps
$U_k(\theta_k)|\psi_{\mathrm{ref}}\rangle$ to
$U_k(\theta_{k+1})|\psi_{\mathrm{ref}}\rangle$: the ansatz structure is
unchanged, only $\theta$ moves.

Actual loop shape:

```python
for index, time_value in enumerate(time_grid):
    evaluation = evaluate_mclachlan_geometry(
        state=state,
        hamiltonian=hamiltonian,
        theta_runtime=theta_current,
        time=time_value,
    )
    fixed_step = solve_fixed_mclachlan_step(
        evaluation.geometry,
        inverse_policy=inverse_policy,
    )

    if index + 1 < len(time_grid):
        dt = time_grid[index + 1] - time_value
        integration = integrate_theta_step(
            theta=theta_current,
            t=time_value,
            dt=dt,
            rhs=theta_dot_rhs,
            method=integrator_method,
        )
        theta_next = integration.theta_next

    points.append(FixedTrajectoryPoint(...))
    theta_current = theta_next
```

> **File.** `trajectory.py`
>
> **Function.**
>
> ```text
> run_fixed_mclachlan_trajectory(...)
> ```
>
> **Objects.** `FixedTrajectoryPoint`, `FixedMclachlanTrajectory`.
>
> **Metadata.**
>
> ```text
> uses_reference_for_decision=False
> uses_future_exact_forecast_for_decision=False
> ```

Pedagogical check: the trajectory runner is a loop over already-understood
single-time-point objects. Do not read it as a new mathematical principle. Read
it as bookkeeping around Step 3 through Step 8.

# Step 10. Runner Boundary from Artifact to JSON Payload

The external runner wraps the previous steps and writes a JSON-friendly payload.
It is the best end-to-end entry point for the implemented fixed-support route.

Code anchor:

```python
payload = run_fixed_ap_mclachlan_from_artifact(
    artifact_json=...,
    times=...,
    integrator_method=...,
    pinv_rcond=...,
    ridge_lambda=...,
)
```

It returns:

```text
state
hamiltonian
trajectory
plot_rows
summary
decision_data_flow
```

> **File.**
>
> ```text
> pipelines/time_dynamics/runners/
>   ap_fixed_from_adapt_artifact.py
> ```
>
> **Functions.**
>
> ```text
> run_fixed_ap_mclachlan_from_runtime_input(...)
> run_fixed_ap_mclachlan_from_artifact(...)
> ```

Reader cue: if a PDF, JSON, or plot row says "fixed AP-McLachlan" in this fresh
implementation, it likely came through this runner boundary. The online support
editing layer is not implied merely because support-patch scoring exists.

# Step 11. Score a Support Patch

After fixed-support geometry is understood, AP adds the support-edit question.
At time point $k$, the controller may choose a patch $\nu_k=(B_k^-,R_k^+)$. The
fresh scorer represents this as:

```python
patch = SupportPatch(
    removed_runtime_indices=(...),
    inserted_count=...,
    inserted_labels=(...),
)
```

The four cases are:

$$
\begin{array}{c|c|c}
B_k^- & R_k^+ & \text{code kind}\\
\hline
\varnothing & \varnothing & \texttt{no\_edit}\\
\varnothing & \ne\varnothing & \texttt{insert}\\
\ne\varnothing & \varnothing & \texttt{delete}\\
\ne\varnothing & \ne\varnothing & \texttt{exchange}
\end{array}
$$

> **File.** `support_patch.py`
>
> **Objects.** `SupportPatch`, `SupportPatchGeometry`, `SupportPatchScore`.
>
> **Constants.** `PATCH_NO_EDIT`, `PATCH_INSERT`, `PATCH_DELETE`,
> `PATCH_EXCHANGE`.

The important code-reading point is that a patch is not the same thing as a
committed controller action. A patch object says "remove these coordinates and
insert this many new coordinates." A controller still has to decide whether that
proposal is safe, useful, and allowed at this time point.

# Step 12. Build Before/After Patch Geometry

The scorer consumes arrays that are already estimated:

```python
geometry = SupportPatchGeometry(
    K_before=K_before,
    f_before=f_before,
    norm_b_sq=V,
    K_insert_cross=B,
    K_insert_insert=C,
    f_insert=q,
)
```

The symbol map is compact: `K_before` is $K_{k,J_kJ_k}$, `f_before` is
$f_{k,J_k}$, and `norm_b_sq` is $V_k$. For insert/exchange patches,
`K_insert_cross` is $B_{r,k}=\Re(\bar T_k^\dagger\bar U_{r,k})$,
`K_insert_insert` is $C_{r,k}=\Re(\bar U_{r,k}^\dagger\bar U_{r,k})$, and
`f_insert` is $q_{r,k}=\Re(\bar U_{r,k}^\dagger\bar b_k)$.

```text
K_insert_cross   -> B_{r,k}
K_insert_insert  -> C_{r,k}
f_insert         -> q_{r,k}
```

`build_after_geometry(...)` removes deleted runtime indices, appends inserted
coordinates, and returns the after-patch matrix/vector. This is one of the most
important transformations in the AP code because it literally changes the local
linear algebra problem.

Transformation pseudo-code:

```python
removed = valid_removed_indices(patch.removed_runtime_indices)
keep = [i for i in range(n_before) if i not in removed]
m = patch.inserted_count

if m == 0:
    K_after = K_before[keep, keep]
    f_after = f_before[keep]
else:
    K_after = zeros((len(keep) + m, len(keep) + m))
    K_after[:len(keep), :len(keep)] = K_before[keep, keep]
    K_after[:len(keep), len(keep):] = K_cross[keep, :]
    K_after[len(keep):, :len(keep)] = K_cross[keep, :].T
    K_after[len(keep):, len(keep):] = K_insert
    f_after = concat(f_before[keep], f_insert)
```

Teaching translation: the scorer is not re-running the whole simulation. It is
asking how the local tangent least-squares geometry would change if the support
were edited.

# Step 13. Compute Before Gain, After Gain, and Patch Score

For any support $S$, the explained drift is
$\Gamma_k^K(S)=f_{k,S}^{\top}K_{k,SS}^{\oplus}f_{k,S}$. The scorer compares
$\Gamma_{\mathrm{before}}=\Gamma_k^K(J_k)$ with
$\Gamma_{\mathrm{after}}=\Gamma_k^K(J_k^\nu)$ and reports
$\Delta_{\nu,k}^{\mathrm{patch}}
=(\Gamma_{\mathrm{after}}-\Gamma_{\mathrm{before}})/(V_k+\varepsilon)$.

Actual scoring flow:

```python
K_after, f_after, keep = build_after_geometry(geometry, patch)

before_gain = gain_for_support(
    matrix=K_before,
    f_vec=f_before,
    indices=range(n_before),
)
after_gain = gain_for_support(
    matrix=K_after,
    f_vec=f_after,
    indices=range(len(f_after)),
)

signed_delta = after_gain - before_gain
denom = norm_b_sq + epsilon
normalized_score = signed_delta / denom
insertion_gain = max(0.0, signed_delta) / denom
deletion_loss = max(0.0, -signed_delta) / denom
```

> **File.** `support_patch.py`
>
> **Functions.** `gain_for_support(...)`, `score_support_patch(...)`,
> `score_patch_payload(...)`.

If the signed delta is positive, the after support explains more drift. If it is
negative, the after support explains less drift. The normalized score divides by
$V_k+\varepsilon$ so scores are interpretable across time points with different
drift magnitudes.

# Step 14. Read Append Gain in Code Terms

For pure append, $B_k^-=\varnothing$ and $R_k^+=\{r\}$. The Chapter 17A gain is
$\rho_{r,k}^{\mathrm{gain}}
=\left[\Gamma_k^K(J_k\cup I_{r,k}^+)-\Gamma_k^K(J_k)\right]_+/(V_k+\varepsilon)$.
In code this is:

```python
score.patch_kind == "insert"
score.insertion_gain
```

The candidate is zero-initialized:
$U_k^{+r}(\theta_k,0)|\psi_{\mathrm{ref}}\rangle
=U_k(\theta_k)|\psi_{\mathrm{ref}}\rangle$. That is why append changes the
tangent plane without jumping the state. The superscript $+r$ means that the
ansatz structure has been augmented by candidate block $r$.

Code-reading rule: append is a tangent-space opportunity test. It asks whether a
new zero-amplitude direction would help fit the current Schrodinger drift.

The implemented append-first path in `adaptive_trajectory.py` is more concrete
than a bare score. It tests candidates by actually building an augmented state,
zero-extending the runtime vector, evaluating augmented geometry, then slicing
that geometry into before/cross/insert blocks.

Append transformation pseudo-code:

```python
if base_step.residual_ratio < residual_threshold:
    return no_edit("residual_below_threshold")

candidates = _append_candidates(
    state=state,
    max_candidates=max_append_candidates,
    allow_incomplete=allow_incomplete_candidate_pool,
)
for candidate in candidates:
    appended_state = state_with_appended_terms(
        state,
        (candidate,),
        theta_runtime=theta_current,
    )
    theta_aug = appended_state.theta_runtime
    n_before = state.runtime_parameter_count
    m_insert = len(theta_aug) - n_before

    evaluation = evaluate_mclachlan_geometry(
        state=appended_state,
        hamiltonian=hamiltonian,
        theta_runtime=theta_aug,
        time=t,
    )
    K, f = evaluation.geometry.K, evaluation.geometry.f
    patch_geometry = SupportPatchGeometry(
        K_before=K[:n_before, :n_before],
        f_before=f[:n_before],
        norm_b_sq=evaluation.geometry.norm_b_sq,
        K_insert_cross=K[:n_before, n_before:],
        K_insert_insert=K[n_before:, n_before:],
        f_insert=f[n_before:],
    )
    score = score_support_patch(
        geometry=patch_geometry,
        patch=SupportPatch(inserted_count=m_insert),
    )
    keep_best_candidate_by(score.rank_score)

if best.insertion_gain >= append_gain_threshold:
    state = best.appended_state
    theta = best.theta_aug
    fixed_step = best.fixed_step
```

# Step 15. Read Prune Loss in Code Terms

For pure prune, $B_k^-=\{b\}$ and $R_k^+=\varnothing$. The tangent-space deletion
loss is $L_{b,k}^{\mathrm{full}}
=\left[\Gamma_k^K(J_k)-\Gamma_k^K(J_k\setminus I_{b,k})\right]_+/(V_k+\varepsilon)$.
In code this is:

```python
score.patch_kind == "delete"
score.deletion_loss
```

This is only the linear/tangent-space screen. Chapter 17A prune acceptance also
requires reduced-state projection, ray safety, differential miss control, shadow
observable safety, persistence, cooldown, and rollback.

> **Implemented now.** Tangent-space deletion-loss scoring.
>
> **Controller layer.** Nonlinear projection and commit/rollback protocol.

Code-reading rule: a small `deletion_loss` says the current tangent fit would not
lose much explained drift if the block disappeared. It does not by itself prove
that removing the block is safe for the nonlinear state.

# Step 16. Read Exchange in Code Terms

For exchange, $B_k^-\ne\varnothing$ and $R_k^+\ne\varnothing$. The after support
is $J_k^\nu=(J_k\setminus I_{\nu,k}^-)\cup I_{\nu,k}^+$, and the combined score
is still $\Delta_{\nu,k}^{\mathrm{patch}}
=(\Gamma_k^K(J_k^\nu)-\Gamma_k^K(J_k))/(V_k+\varepsilon)$.

In code:

```python
score.patch_kind == "exchange"
score.normalized_score
```

Exchange should be read conservatively. Each inserted block still needs append
gain/confirmation, and each removed block still needs prune safety. The combined
score is a recomputation on the final proposed support, not a proof that the
delete and insert are separately harmless.

# Step 17. Current Append-First Adaptive Loop

The currently implemented adaptive trajectory is append-first. The route-level
event map it approximates is
$U_k(\theta_k)|\psi_{\mathrm{ref}}\rangle
\xrightarrow{\mathrm{diagnose/select}}\nu_k^\star
\xrightarrow{\mathsf S_{\nu_k^\star}}
U_k^\nu(\theta_k^{\nu,+})|\psi_{\mathrm{ref}}\rangle
\xrightarrow{\operatorname{Step}_{\mathrm{declared}}}
U_{k+1}(\theta_{k+1})|\psi_{\mathrm{ref}}\rangle$.

Actual loop shape:

```python
current_state = state
theta_current = state.theta_runtime
append_count = 0

for k, t in enumerate(time_points):
    evaluation = evaluate_mclachlan_geometry(
        state=current_state,
        hamiltonian=hamiltonian,
        theta_runtime=theta_current,
        time=t,
    )
    fixed_step = solve_fixed_mclachlan_step(
        evaluation.geometry,
        inverse_policy=inverse_policy,
    )

    if can_still_append(k, append_count):
        decision, new_state, new_theta, new_eval, new_step = (
            _select_append_patch(...)
        )
        if decision.accepted:
            current_state = new_state
            theta_current = new_theta
            evaluation = new_eval
            fixed_step = new_step
            append_count += 1

    if k + 1 < len(time_points):
        state_for_rhs = current_state
        theta_next = integrate_theta_step(
            theta=theta_current,
            t=t,
            dt=time_points[k + 1] - t,
            rhs=lambda th, ts: solve_rhs(state_for_rhs, th, ts),
            method=integrator_method,
        ).theta_next

    points.append(AdaptiveTrajectoryPoint(...))
    theta_current = theta_next
```

The key mutation is `current_state = new_state`: appending is a structural edit
to the support and layout. The matching theta mutation is `theta_current =
new_theta`, where the old coordinates are preserved and new coordinates start at
zero.

# Exact-Reference Boundary

Strict AP-McLachlan decisions use measurement-compatible data for the prepared
ansatz/circuit state. Exact references may appear only as diagnostic outputs:
plots, errors, spectra, overlays, and post-hoc audits. Forbidden decision inputs
include exact target states, exact future trajectories, exact future
fidelity/error, and exact forecast helpers.

Code reading anchor:

```python
"decision_data_flow": {
    "uses_reference_for_decision": False,
    "uses_exact_reference_for_decision": False,
    "uses_future_exact_forecast_for_decision":
        False,
    "uses_statevector_as_ideal_observable_estimator": True,
}
```

This appears in the fixed artifact runner payload. It is a useful boundary
marker: local statevector evaluation is used as an ideal observable estimator for
the prepared state, not as an exact target-trajectory oracle for decisions.

# A Concrete Reading Pass

When you open the code, read it in this order.

1. Start with `state.py`. Confirm where `terms`, `layout`, `theta_runtime`, and
   `psi_ref` enter the AP state. This is the ordered ansatz $U_k$, the runtime
   vector $\theta_k$, and the reference state.
2. Move to `hamiltonian.py`. Confirm how `matrix_at(time)` turns the static
   polynomial plus drive data into $H(t_k)$.
3. Open `geometry_eval.py`. Find the state preparation call, then find
   horizontalization. This is where $|\psi_k\rangle$, $\bar T_k$, and $\bar b_k$
   are built.
4. Open `geometry.py`. Treat this as the typed container for $K_k$, $f_k$, and
   $V_k$.
5. Open `inverse.py` and `fixed_step.py`. These are the solve layer:
   $K^\oplus f$, explained drift, and residual ratio.
6. Open `integrators.py` and `trajectory.py`. These turn a single-time-point
   solve into a path over the time grid.
7. Open `support_patch.py`. It scores how a hypothetical support edit would
   change the local McLachlan geometry.
8. Open `adaptive_trajectory.py` last. It shows the implemented append-first
   controller path that mutates state/layout/theta.

The most common confusion is to read `support_patch.py` first because AP means
append-prune. In execution order, that is backwards. The patch scorer needs the
geometry produced by the fixed-support step.

# Reading the Fixed Runner as a Call Stack

The fixed artifact runner is a good first code path because it has no support
mutation. Its job is to make the AP objects, run the fixed-support trajectory,
and emit JSON-shaped evidence.

Compressed actual call shape:

```python
def run_fixed_ap_mclachlan_from_runtime_input(
    runtime_input,
    times,
    integrator_method="euler",
    pinv_rcond=1.0e-10,
    ridge_lambda=0.0,
    enable_drive=False,
    drive_config=None,
):
    state = state_from_scaffold_runtime_input(runtime_input)
    hamiltonian = time_dependent_hamiltonian_from_runtime_input(
        runtime_input,
        drive_config=(drive_config if enable_drive else None),
    )
    inverse_policy = McLachlanInversePolicy(
        pinv_rcond=pinv_rcond,
        ridge_lambda=ridge_lambda,
    )
    trajectory = run_fixed_mclachlan_trajectory(
        state=state,
        hamiltonian=hamiltonian,
        times=times,
        inverse_policy=inverse_policy,
        integrator_method=integrator_method,
    )
    rows = _plot_rows(trajectory)
    return {
        "state": state.to_json_dict(),
        "hamiltonian": hamiltonian.to_json_dict(),
        "trajectory": trajectory.to_json_dict(),
        "plot_rows": rows,
        "summary": _summary_from_rows(rows, ...),
        "decision_data_flow": {...},
    }
```

Read this function as five conversions:

1. Static ansatz input becomes `APMcLachlanState`.
2. Static plus drive information becomes `TimeDependentHamiltonian`.
3. Numeric inverse knobs become `McLachlanInversePolicy`.
4. State plus Hamiltonian plus time grid becomes a trajectory.
5. Trajectory becomes report-friendly rows, summary, and decision metadata.

This runner does not pick candidate support edits. It is still AP-McLachlan code
because the fixed-support McLachlan solve is the geometric core that append and
prune decisions use.

# Geometry Evaluation as Array Construction

`evaluate_mclachlan_geometry(...)` is the central code-math translator. It turns
a state and a time into arrays with the same roles as Chapter 17A symbols.

The live variables have these shapes:

```text
theta_runtime        shape (N,)
psi                  shape (D,)
hmat                 shape (D, D)
h_psi                shape (D,)
b_bar                shape (D,)
tangent_columns      N vectors, each shape (D,)
tangent_matrix/Tbar  shape (D, N)
K                    shape (N, N)
f                    shape (N,)
norm_b_sq/V          scalar
```

The construction order is:

```python
theta = theta_runtime or state.theta_runtime
indices = runtime_indices or range(state.runtime_parameter_count)

psi, tangents = prepare_state_with_runtime_tangents(
    theta,
    state.psi_ref,
    runtime_indices=indices,
)
psi = normalize(psi)

hmat = hamiltonian.matrix_at(time)
h_psi = hmat @ psi
energy = real(inner(psi, h_psi))
b_bar = -1j * (h_psi - energy * psi)

for idx in indices:
    raw_tangent = tangents[idx]
    horizontal = raw_tangent - psi * inner(psi, raw_tangent)
    columns.append(horizontal)

Tbar = column_stack(columns)
K = real(Tbar.conj().T @ Tbar)
f = real(Tbar.conj().T @ b_bar)
V = real(inner(b_bar, b_bar))
```

The most important line for the math is the Gram construction
`K = real(Tbar.conj().T @ Tbar)`. It says that $K_k$ measures tangent directions
against each other after horizontalization. The line
`f = real(Tbar.conj().T @ b_bar)` says that $f_k$ measures each tangent direction
against the Schrodinger drift.

When `runtime_indices` is a subset, the evaluator builds a restricted geometry.
That is useful because support-patch scoring and diagnostics often ask what a
smaller or altered support would explain.

# What the Normal Solve Removes

The least-squares problem begins with a velocity vector $u$:
$\min_u\|\bar T_ku-\bar b_k\|^2$. The code does not store every possible $u$.
It solves for the best one and then stores only the result plus diagnostics.

Expansion:

$$
\|\bar T_ku-\bar b_k\|^2
=u^\top K_ku-2u^\top f_k+V_k.
$$

Stationarity gives $K_ku=f_k$. The supported inverse gives
$u^\star=K_k^\oplus f_k$. The optimized explained length is
$f_k^\top u^\star=f_k^\top K_k^\oplus f_k$. That is why `gamma` appears in the
code: the optimized velocity has already been substituted back into the score.

Implementation reading:

```python
solve = solve_theta_dot(geometry.K, geometry.f)
theta_dot = solve.theta_dot
gamma = solve.gamma
residual_sq = geometry.norm_b_sq - gamma
residual_ratio = residual_sq / (geometry.norm_b_sq + epsilon)
```

If `gamma` is close to `norm_b_sq`, the tangent span explains almost all of the
instantaneous drift. If `residual_ratio` is large, the current support is missing
important tangent directions. The append-first controller uses that residual as
one gate before it spends work scoring append candidates.

# Time Integration Recomputes Geometry

The integrator is easy to underestimate. It receives an RHS function, and that
RHS function re-evaluates McLachlan geometry at the staged theta and time. For
Euler this happens once. For RK4 it happens four times.

Fixed trajectory RHS:

```python
def theta_dot_rhs(theta_value, time_value):
    evaluation = evaluate_mclachlan_geometry(
        state=state,
        hamiltonian=hamiltonian,
        theta_runtime=theta_value,
        time=time_value,
    )
    step = solve_fixed_mclachlan_step(
        evaluation.geometry,
        inverse_policy=inverse_policy,
    )
    return step.theta_dot
```

This means the trajectory is not simply reusing the first $\dot\theta_k$ across
the whole step. The RHS asks again: at this staged parameter vector and staged
time, what tangent fit does McLachlan prescribe?

For the append-first path, the RHS closes over `state_for_rhs = current_state`.
That detail matters. If an append is accepted before integration, RK stages use
the appended support. If no append is accepted, RK stages use the old support.

# State Mutation by Append

`state_with_appended_terms(...)` is the clearest current example of support
mutation. It does four things in sequence:

1. It checks that the incoming theta has the old runtime length.
2. It creates `terms = old_terms + appended_terms`.
3. It rebuilds the parameter layout for the longer term list.
4. It allocates a longer theta vector, copies old coordinates into the prefix,
   and leaves new coordinates at zero.

Code shape:

```python
theta_base = theta_runtime or state.theta_runtime
assert len(theta_base) == state.runtime_parameter_count

terms = tuple(state.terms) + tuple(appended_terms)
layout = build_parameter_layout(terms, ...)

prefix_runtime_count = layout.blocks[len(state.terms)-1].runtime_stop
assert prefix_runtime_count == state.runtime_parameter_count

theta = zeros(layout.runtime_parameter_count)
theta[:len(theta_base)] = theta_base

executor = _executor_for_terms(terms, layout)
return APMcLachlanState(
    terms=terms,
    layout=layout,
    theta_runtime=theta,
    psi_ref=state.psi_ref,
    psi_initial=state.prepare_state(theta_base),
    executor=executor,
    ...
)
```

Mathematically, this is the zero-initialized append condition:
$U^+(\theta,0)|\psi_{\mathrm{ref}}\rangle
=U(\theta)|\psi_{\mathrm{ref}}\rangle$. The state does not jump at the instant
of append because the new coordinates start at zero. What changes immediately
is the tangent space available for the next McLachlan solve.

The `psi_initial=state.prepare_state(theta_base)` field records the prepared
state at the moment the augmented ansatz state is created. That is useful
provenance for the appended manifold.

# Append Candidate Gate by Gate

`_select_append_patch(...)` is the current controller decision kernel. It returns
five things: a `PatchDecision`, maybe a new state, maybe a new theta vector,
maybe a new geometry evaluation, and maybe a new fixed step.

Gate 1: residual trigger.

```python
if base_step.residual_ratio < residual_ratio_threshold:
    return no_edit("residual_below_threshold")
```

The controller only considers appending when the current support leaves enough
unexplained drift.

Gate 2: candidate availability.

```python
candidates = _append_candidates(
    state,
    max_candidates=max_append_candidates,
    allow_incomplete=allow_incomplete_candidate_pool,
)
if not candidates:
    return no_edit("no_append_candidates")
```

The candidate list comes from `state.candidate_pool_terms`, excluding already
selected labels. If `allow_incomplete_candidate_pool=False`, the state must say
the candidate pool is complete.

Gate 3: score each candidate by augmented geometry.

```python
for candidate in candidates:
    appended_state = state_with_appended_terms(state, (candidate,))
    theta_aug = appended_state.theta_runtime
    n_before = state.runtime_parameter_count
    m_insert = len(theta_aug) - n_before

    evaluation = evaluate_mclachlan_geometry(
        state=appended_state,
        theta_runtime=theta_aug,
        time=time,
    )
    K = evaluation.geometry.K
    f = evaluation.geometry.f
```

The candidate is scored in the geometry of the appended state, not merely by a
label lookup. This is why the pseudo-code must show a real state alteration.

Gate 4: slice augmented geometry into before and insert blocks.

```python
patch_geometry = SupportPatchGeometry(
    K_before=K[:n_before, :n_before],
    f_before=f[:n_before],
    norm_b_sq=evaluation.geometry.norm_b_sq,
    K_insert_cross=K[:n_before, n_before:],
    K_insert_insert=K[n_before:, n_before:],
    f_insert=f[n_before:],
)
score = score_support_patch(
    geometry=patch_geometry,
    patch=SupportPatch(inserted_count=m_insert),
)
```

This is the most important slice pattern in the append code. The full augmented
geometry is partitioned into old-old, old-new, new-new, old force, and new force
pieces.

Gate 5: choose and accept.

```python
if best is None:
    return no_edit("no_finite_append_score")

if best.score.insertion_gain < append_gain_threshold:
    return no_edit("append_gain_below_threshold")

return accepted_insert(
    state=best.appended_state,
    theta=best.theta_aug,
    evaluation=best.evaluation,
    step=best.step,
)
```

Only an accepted append mutates the trajectory state. Rejected candidates still
produce useful diagnostic scores, but the support remains unchanged.

# Before/After Block Algebra

Support-patch scoring can be read as block linear algebra. Start with the
before-support arrays $K_{JJ}$ and $f_J$. A delete patch keeps a subset
$J\setminus B^-$. An insert patch appends new coordinates $R^+$. An exchange
does both.

For an insert or exchange, the after matrix has block form:

$$
K_{\mathrm{after}}=
\begin{bmatrix}
K_{\mathrm{keep,keep}} & K_{\mathrm{keep,new}}\\
K_{\mathrm{new,keep}} & K_{\mathrm{new,new}}
\end{bmatrix},
\qquad
f_{\mathrm{after}}=
\begin{bmatrix}f_{\mathrm{keep}}\\ f_{\mathrm{new}}\end{bmatrix}.
$$

The code builds exactly that:

```python
K_after = zeros((keep_len + m, keep_len + m))
K_after[:keep_len, :keep_len] = K_before[keep, keep]
K_after[:keep_len, keep_len:] = K_cross[keep, :]
K_after[keep_len:, :keep_len] = K_cross[keep, :].T
K_after[keep_len:, keep_len:] = K_insert

f_after = concat(f_before[keep], f_insert)
```

Then both before and after supports are scored with the same supported-inverse
convention. That shared inverse policy is important: patch comparisons would be
meaningless if before and after used different rank/ridge rules.

# What Each Recorded Point Means

The fixed trajectory stores `FixedTrajectoryPoint` objects. Each point records:

```text
index
time
theta_runtime
energy_expectation
geometry
fixed_step
integration_to_next
```

This point is a snapshot before moving to the next time value. The `theta_runtime`
field is the theta used to evaluate that point. If there is a next time point,
`integration_to_next` records how theta was advanced after the point was
evaluated.

The append-first trajectory stores `AdaptiveTrajectoryPoint` objects. Each point
adds:

```text
runtime_parameter_count
logical_parameter_count
patch_decision
```

Those counts are useful because append changes the size of the runtime vector.
If an append is accepted at a point, the recorded count can increase relative to
the previous point.

The `patch_decision` object tells you why the controller did or did not append:

```text
residual_below_threshold
no_append_candidates
no_finite_append_score
append_gain_below_threshold
accepted_best_append_gain
append_not_considered
```

Those strings are learning handles. When a trajectory does not append, read the
reason before looking at the math.

# Configuration Knobs in Plain Terms

`McLachlanInversePolicy.pinv_rcond` controls which eigenmodes of $K$ count as
supported. Larger values drop more near-null modes. Smaller values retain more
modes and may amplify numerical noise.

`McLachlanInversePolicy.ridge_lambda` adds a diagonal ridge before the inverse.
It stabilizes the solve by lifting small eigenvalues, but it also changes the
metric being inverted.

`AppendControllerConfig.max_append_candidates` caps how many candidates are
scored at each time point.

`AppendControllerConfig.max_total_appends` caps how many append commits can
occur across the trajectory.

`AppendControllerConfig.residual_ratio_threshold` is the trigger threshold. If
the current fixed support already explains the drift well enough, append
candidates are skipped.

`AppendControllerConfig.append_gain_threshold` is the acceptance threshold after
candidate scoring. The best candidate still has to clear this gain floor.

`allow_incomplete_candidate_pool` controls whether an incomplete candidate pool
can still be used for append attempts. For a learning run this may be useful; for
paper-facing semantics it should be read carefully with provenance.

# Current Code Versus Full 17A

The current implementation covers the backbone of Chapter 17A:

```text
state preparation
horizontal tangent geometry
Schrodinger drift
supported inverse solve
fixed-support trajectory
support-patch scoring
append-first support mutation
decision telemetry
```

The current implementation does not yet provide the full prune/exchange commit
protocol described by active 17A. In particular, deletion scoring exists, but
the nonlinear reduced-state projection, ray-safety checks, shadow observable
guards, persistence logic, cooldown, and rollback are still controller-layer
work.

This distinction helps you read evidence. A support-patch score can say "this
delete would lose little tangent explained drift." It cannot yet say "the code
has safely pruned and committed this block through the full 17A safety protocol."

# Debugging by Symptom

If you see a theta length mismatch, start in `state.py`. The likely issue is a
layout/runtime-vector disagreement.

If geometry evaluation rejects an index, inspect `runtime_indices`. Indices must
be unique and inside the current runtime parameter count.

If `K_insert_cross` shape errors occur, check the append partition:
`K[:n_before, n_before:]` must have shape `(n_before, m_insert)`.

If no append is considered, inspect `residual_ratio_threshold`,
`max_total_appends`, and whether the time point is the final point.

If there are no candidates, inspect `candidate_pool_terms`, selected labels, and
`allow_incomplete_candidate_pool`.

If a candidate scores but is rejected, inspect `insertion_gain` against
`append_gain_threshold`.

If `rank_score` is `None`, the gain solve likely failed or produced a nonfinite
value. Start with the before/after matrices and inverse policy.

If exact-reference leakage is suspected, inspect `decision_data_flow` and the
trajectory metadata. Decision fields should say reference data and exact future
forecasts are not used for decisions.

# Symbol-to-Code Map

Use this map while reading code and Chapter 17A together.

- $U_k$: current ordered ansatz circuit.
  Code anchor: `APMcLachlanState.terms` and `layout`.
- $U_k(\theta_k)|\psi_{\mathrm{ref}}\rangle$: prepared ansatz state.
  Code anchor: `prepare_state(...)`.
- $\theta_k$: runtime parameters.
  Code anchor: `theta_runtime`.
- $|\psi_{\mathrm{ref}}\rangle$: reference state.
  Code anchor: `psi_ref`.
- $H(t)$: time-dependent Hamiltonian.
  Code anchor: `matrix_at(...)`.
- $\bar T_k$: horizontal tangent matrix.
  Code anchor: geometry evaluator.
- $\bar b_k$: horizontal Schrodinger drift.
  Code anchor: `b_bar`.
- $K_k$: McLachlan normal matrix.
  Code anchor: `McLachlanGeometry.K`.
- $f_k$: force / overlap vector.
  Code anchor: `McLachlanGeometry.f`.
- $V_k$: drift norm squared.
  Code anchor: `McLachlanGeometry.norm_b_sq`.
- $K^{\oplus}$: supported inverse.
  Code anchor: `supported_inverse(...)`.
- $\dot\theta_k$: parameter velocity.
  Code anchor: `McLachlanSolve.theta_dot`.
- $\Gamma_k^K(S)$: explained drift on support $S$.
  Code anchor: `gamma_for_support(...)`.
- $\rho_k$: residual ratio.
  Code anchor: `FixedMcLachlanStep.residual_ratio`.
- $\nu_k$: support patch.
  Code anchor: `SupportPatch`.
- $R_k^+$: inserted block set.
  Code anchor: `inserted_count` and `inserted_labels`.
- $B_k^-$: removed block set.
  Code anchor: `removed_runtime_indices`.
- Append gain: positive patch improvement.
  Code anchor: `insertion_gain`.
- Prune loss: lost explained drift.
  Code anchor: `deletion_loss`.
- Exchange: delete plus insert.
  Code anchor: `PATCH_EXCHANGE`.

# Tests as Mini Lessons

Read these tests as executable examples:

- `test/test_ap_mclachlan_state_hamiltonian.py`: state adapter and Hamiltonian
  provider.
- `test/test_ap_mclachlan_geometry_eval.py`: prepared state, tangents, drift,
  geometry payload.
- `test/test_ap_mclachlan_inverse.py`: supported inverse, retained modes, gamma.
- `test/test_ap_mclachlan_fixed_step.py`: fixed-support Eq. (8) solve and
  residual.
- `test/test_ap_mclachlan_trajectory.py`: propagation over a time grid.
- `test/test_ap_mclachlan_fixed_runner.py`: artifact-to-trajectory payload.
- `test/test_ap_mclachlan_support_patch.py`: no edit, insert, delete, exchange,
  pseudoinverse behavior, cost ranking.
- `test/test_ap_mclachlan_adaptive_trajectory.py`: append-first state mutation
  without exact-reference decisions.

Use the tests as a low-stakes route through the concepts. If a test constructs a
tiny geometry object by hand, read that as the smallest possible version of the
Chapter 17A formula. If a test calls the runner, read it as an end-to-end proof
that the code path can go from artifact-shaped input to trajectory-shaped output.

# One-Pass Mental Model

The code executes the chain ansatz artifact $\to$ AP state $\to H(t)\to
(\bar T,\bar b)\to(K,f,V)\to\dot\theta=K^\oplus f\to
U_k(\theta_{\mathrm{next}})|\psi_{\mathrm{ref}}\rangle\to\nu=(B^-,R^+)$.

The mathematical reason is: fit the Schrodinger drift with the current tangent
span, then edit the support when the tangent fit justifies it.

The implementation reason is:

```text
keep Hamiltonian construction, state preparation, geometry evaluation,
linear solves, integration, support-patch scoring, and controller commits
as separate surfaces.
```

That separation is what lets us check each piece without confusing fixed-support
McLachlan propagation, AP support editing, and exact-reference diagnostics.

# Mutation Ledger

Use this ledger when you are trying to tell whether a function changes the
mathematical object or only measures it.

`state_from_scaffold_runtime_input(...)` constructs the initial AP object. It
does not advance time. It validates the ansatz input, normalizes the reference
states, builds the compiled executor, and freezes the initial ordered ansatz
$U_k$ plus runtime vector $\theta_k$ into `APMcLachlanState`. The function name
contains a legacy repo word; the object it constructs is the AP ansatz state.

`evaluate_mclachlan_geometry(...)` measures local geometry at one state and one
time. It does not mutate `state` or `theta_runtime`. It prepares
$|\psi_k\rangle$, builds horizontal tangent columns, forms $\bar b_k$, then
packages $K_k$, $f_k$, and $V_k$.

`solve_fixed_mclachlan_step(...)` solves an array problem. It does not know
about generators, Hamiltonian families, candidate pools, or exact references. It
turns `geometry.K` and `geometry.f` into $\dot\theta_k$, explained drift
`gamma`, and residual diagnostics.

`integrate_theta_step(...)` mutates only the runtime vector in the mathematical
sense: $\theta_k\mapsto\theta_{k+1}$. The ansatz structure $U_k$ is unchanged.
For RK4, the RHS is called at staged theta/time pairs, so geometry is recomputed
inside those stages.

`build_after_geometry(...)` mutates the local linear algebra problem, not the
actual AP state. It deletes rows/columns for removed runtime indices and appends
insert blocks to form an after-support matrix and force vector.

`score_support_patch(...)` compares two local least-squares explanations:
before support and after support. It reports gain/loss numbers, but it does not
commit the patch.

`state_with_appended_terms(...)` is the first true support mutation in the
implemented append-first path. It rebuilds the layout with new terms, preserves
old theta coordinates, appends zero coordinates for the new block, rebuilds the
executor, and returns a new `APMcLachlanState`.

`run_append_mclachlan_trajectory(...)` is the current adaptive orchestrator. It
evaluates the fixed step, asks `_select_append_patch(...)` whether an append is
worth accepting, swaps in the new state/theta if accepted, then integrates the
current theta to the next time point.

The practical reading rule is: geometry and scoring functions produce evidence;
state construction and append functions change the ansatz state; integrators
change theta; runner functions decide when those pieces happen.

# NumPy Idioms in This Route

These are the NumPy patterns most likely to come up while reading code details.

`np.asarray(x, dtype=float).reshape(-1)` converts input into a one-dimensional
numeric vector. In this route it is used for theta vectors and force vectors.
The `reshape(-1)` part means "flatten this into one dimension."

`np.asarray(x, dtype=complex).reshape(-1)` does the same thing for statevectors
and tangent vectors. These are complex because quantum amplitudes are complex.

`np.linalg.norm(psi)` computes the Euclidean length of a statevector. The code
uses it to normalize states before geometry is evaluated.

`np.vdot(a, b)` computes the complex inner product with conjugation on the first
argument. In the geometry evaluator, `np.vdot(psi, h_psi)` is
$\langle\psi|H|\psi\rangle$.

`hmat @ psi` is matrix-vector multiplication. It applies the Hamiltonian matrix
to the prepared state.

`vector - psi * np.vdot(psi, vector)` subtracts the component of `vector` along
`psi`. This is the horizontal projection used for tangent vectors.

`np.column_stack(columns)` turns a list of same-length vectors into a matrix
whose columns are those vectors. In this document that matrix is $\bar T_k$.

`Tbar.conj().T @ Tbar` means conjugate transpose times matrix. It builds the
complex tangent Gram matrix before taking the real part.

`np.real(...)` discards numerical imaginary roundoff when the intended
McLachlan matrix/vector is real.

`0.5 * (K + K.T)` symmetrizes a real matrix. It guards against tiny numerical
asymmetry in the Gram matrix.

`np.linalg.eigh(K)` diagonalizes a symmetric/Hermitian matrix. The supported
inverse uses it to decide which eigenmodes are trusted.

`keep = abs(eigvals) > threshold` creates a Boolean mask. The inverse keeps only
the eigenmodes whose eigenvalues are large enough.

`inv_eigs[keep] = 1.0 / eigvals[keep]` inverts only retained eigenvalues. Dropped
eigenmodes remain zero in the inverse.

`K_before[np.ix_(keep, keep)]` extracts the square submatrix on retained runtime
indices. It is the safe way to select the same row and column subset.

`K[:n_before, :n_before]` is the old-old block of an augmented matrix.
`K[:n_before, n_before:]` is the old-new cross block. `K[n_before:, n_before:]`
is the new-new insert block.

`np.concatenate((old, new))` joins vectors end to end. In support-patch scoring
it forms the after-patch force vector from kept old forces plus inserted forces.

`np.zeros((m, n))` allocates an array filled with zeros. The append code uses
zeros for new theta coordinates and for assembling after-patch block matrices.

`np.isfinite(x)` checks for non-NaN and non-infinite values. The solver and
scorer use it to fail closed on invalid numerical results.

When reading any NumPy line, ask two questions: what array shape does this line
produce, and which Chapter 17A symbol does that array represent?
